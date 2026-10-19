"""
Pick and place primitives: cuRobo motion plans (plan_*) and checks of the executed result (judge_*).
"""

import os

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import Obstacle
from curobo.wrap.reacher.motion_gen import PoseCostMetric

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import trimesh
from trimesh.creation import icosphere, axis
from disco.utils.conversion_utils import (
    pose_to_transformation, 
    move_pose_along_local_z, 
    move_pose_along_local_x,
    move_pose_along_local_y,
    transformation_to_pose, 
    invert_transformation,
    quaternion_to_matrix
)
from disco.utils.common import to_torch_pose
import copy
from tqdm import tqdm

EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

# Constants for motion planning
POST_GRASP_LIFT = 0.15


def _merge_rows(traj, ok, fb_traj, fb_ok):
    """Replace each failed row of a batched plan with its fallback plan where that one succeeded.

    Args:
        traj: (B, T, 7) primary joint trajectories.
        ok: (B,) bool success of the primary plans.
        fb_traj: (B, T', 7) fallback joint trajectories.
        fb_ok: (B,) bool success of the fallback plans.

    Returns:
        Merged (B, max(T, T'), 7) trajectories (short ones hold their last waypoint) and merged success.
    """
    T = max(traj.shape[1], fb_traj.shape[1])
    pad = lambda x: x if x.shape[1] == T else torch.cat([x, x[:, -1:].repeat(1, T - x.shape[1], 1)], dim=1)
    traj, fb_traj = pad(traj), pad(fb_traj.to(traj.device))
    use = (~ok) & fb_ok.to(ok.device)
    traj = torch.where(use[:, None, None], fb_traj, traj)
    return traj, ok | use
GRASP_DEPTH = 0.12
OPEN = 1
CLOSED = -1

from disco.motion.motionplanner import CUROBO_MESH_DIR, MotionPlanner
# from disco.envs.shelf_packing_env import ShelfPackingMultiObject

class ActionPrimitives(MotionPlanner):
    """
    Action primitives for robot manipulation using cuRobo motion planning.
    Inherits from MotionPlanner to access all motion planning functionality.
    """
    
    def __init__(
        self,
        env,           # TODO: Make this more general, to be able to take in any environment.
        movement_threshold: float = 0.01  # threshold for detecting unwanted object movement (meters)
    ):
        """Export each object's collision mesh and build the cuRobo planner and pose constraints.

        Args:
            env: Shelf or stacking ManiSkill env (32 slots) that owns these primitives.
            movement_threshold: Largest object pose change (m) that Move tolerates.
        """

        self.env = env
        self.num_envs = env.num_envs
        self.movement_threshold = movement_threshold
        
        planner_initial_object_poses = {}
        individual_mesh_files = {}

        # TODO: How does actor.name deal with repeated actors? Need to debug this case
        for actor in self.env.actors:
            mesh = actor.get_collision_meshes()[0]            # Initialize with env 0
            # Store each mesh individually with actor name
            mesh_name = f"{actor.name}"
            os.makedirs(CUROBO_MESH_DIR, exist_ok=True)
            mesh_path = os.path.join(CUROBO_MESH_DIR, f"{mesh_name}.obj")
            mesh.export(mesh_path)
            individual_mesh_files[mesh_name] = mesh_path
            initial_object_pose = self.env.get_object_pose(actor)[0].cpu().detach().numpy()
            planner_initial_object_poses[mesh_name] = initial_object_pose

        # Initialize parent MotionPlanner class
        super().__init__(
            mesh_file_dict=individual_mesh_files,
            initial_object_poses=planner_initial_object_poses,
        )

        # Useful for action primitives:

        self.lift_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 1, 1, 0], device="cuda:0"),
            project_to_goal_frame=False
        )
        self.lift_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.lift_constraint,          
        )

        self.along_x_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 0, 1, 1], device="cuda:0"),
            project_to_goal_frame=True
        )
        self.along_x_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.along_x_constraint,          
        )
        
        self.along_y_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 1, 0, 1], device="cuda:0"),
            project_to_goal_frame=True
        )
        self.along_y_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.along_y_constraint,          
        )

        self.along_z_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 1, 1, 0], device="cuda:0"),
            project_to_goal_frame=True
        )
        self.along_z_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.along_z_constraint,          
        )

        self.placement_constraint = PoseCostMetric(
            reach_vec_weight = torch.tensor([1, 0, 0, 1, 1, 1], device="cuda:0"),
            project_to_goal_frame=True
        )
        self.placement_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.placement_constraint,          
        )

        # Stacking placement with free yaw: reach the goal position and roll / pitch exactly, but any
        # rotation about WORLD z (the object lands on the same face at any yaw).
        self.stack_place_constraint = PoseCostMetric(
            reach_vec_weight = torch.tensor([1, 1, 0, 1, 1, 1], device="cuda:0"),
            project_to_goal_frame=False
        )
        self.stack_place_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.stack_place_constraint,
        )
    
    
    # =============================== COLLISION WORLD FUNCTIONS =============================== #
    
    def compute_pose_distance(self, pose1: torch.Tensor, pose2: torch.Tensor) -> torch.Tensor:
        """Return the largest per-element difference between two poses, ignoring quaternion sign.

        Args:
            pose1: (..., 7) poses as (x, y, z, qw, qx, qy, qz).
            pose2: (..., 7) poses in the same format.

        Returns:
            (...,) max absolute difference over the 7 elements.
        """
        # Position difference (first 3 elements)
        pos_diff = torch.abs(pose1[..., :3] - pose2[..., :3])
        
        # Quaternion difference (last 4 elements)
        # Handle quaternion sign ambiguity: q and -q represent the same rotation
        q1 = pose1[..., 3:]
        q2 = pose2[..., 3:]
        q_diff1 = torch.abs(q1 - q2)
        q_diff2 = torch.abs(q1 - (-q2))  # Check negative quaternion
        q_diff = torch.minimum(q_diff1, q_diff2)
        
        # Concatenate position and quaternion differences
        diff = torch.cat([pos_diff, q_diff], dim=-1)  # (..., 7)
        
        # Take max across all 7 elements
        pose_distance = diff.max(dim=-1)[0]
        
        return pose_distance
    
    def update_collision_world(self, env_idx: int = 0):
        """Move each movable object's cuRobo obstacle to its current pose in one sim slot.

        Args:
            env_idx: Sim slot to read object poses from.
        """

        if not hasattr(self.env, 'movable_actors'):
            print("movable_actors list not found. Please create it first.")
            return

        for actor in self.env.movable_actors:

            obj_pose = self.env.get_object_pose(actor)[env_idx]          
            curobo_pose = obj_pose.clone().cpu().detach().numpy()

            world_to_object_transform = pose_to_transformation(curobo_pose, format='wxyz')
            initial_to_object_transform = pose_to_transformation(self.initial_object_poses[actor.name], format='wxyz')
            object_transform = world_to_object_transform @ invert_transformation(initial_to_object_transform)

            obj_pose = to_torch_pose(transformation_to_pose(object_transform, format='wxyz'))
            curobo_pose = obj_pose.clone()
            curobo_pose[..., 0] = curobo_pose[..., 0] + 0.615

            pose = Pose(
                position=curobo_pose[..., :3],
                quaternion=curobo_pose[..., 3:]
            )

            # IMPORTANT: Use update_obstacle_pose to update GPU collision tensors (_cube_tensor_list)
            # update_obstacle_pose_in_world_model only updates CPU reference, not collision tensors
            self.motion_gen.world_coll_checker.update_obstacle_pose(
                name=actor.name,
                w_obj_pose=pose,
                update_cpu_reference=True  # Also update CPU reference for consistency
            )
        
    def visualize_planner_world(self, pose=None, env_idx: int = 0):
        """Show the cuRobo collision world and robot spheres at one sim slot's current state.

        Args:
            pose: Optional (7,) pose in the robot frame, drawn as an axis frame.
            env_idx: Sim slot to read robot and object state from.
        """
        qpos = self.env.get_robot_qpos()
        self.update_collision_world(env_idx)
        current_joint_state = qpos[env_idx, :7].to("cuda:0").to(torch.float32)
        # Check if planner is initialized
        # if self.planner is None:
        #     print("Planner not initialized. Call initialize_planner() first.")
        #     return
        self.visualize_world_and_robot(current_joint_state, pose)
    
    def attach_object_to_robot(self, object_name):
        """Attach an object to the planner's gripper and disable its world obstacle.

        Uses the robot joints of sim slot 0.

        Args:
            object_name: Name of the held object.
        """
            
        # Get current robot state
        robot_qpos = self.env.get_robot_qpos().to("cuda:0")

        current_joint_state = robot_qpos[0, :7]
        
        # Attach the object
        success = self.attach_objects_to_robot(
            current_joint_state=current_joint_state,
            object_names=[object_name]
        )        
        self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=object_name)
                
    def detach_object_from_robot(self, object_name):
        """Detach all objects from the planner's gripper and re-enable one world obstacle.

        Args:
            object_name: Object whose world obstacle is re-enabled.
        """
        
        success = self.detach_objects_from_robot()
        self.motion_gen.world_coll_checker.enable_obstacle(enable=True, name=object_name)
    
    # =============================== ACTION PRIMITIVES =============================== #

    def OpenGripper(
        self
        ):
        """Open the gripper in every sim slot without moving the arm.

        Returns:
            (num_envs, 1, 8) one-step trajectories and an all-True (num_envs,) success array.
        """
        # Get current joint state (num_envs, 7)
        current_joint_state = self.env.get_joint_state().to("cuda:0")
        
        # Create single-step trajectory (num_envs, 1, 7)
        trajectories = current_joint_state.unsqueeze(1).cpu().detach().numpy()
        
        # Add OPEN gripper state (num_envs, 1, 1)
        gripper_state = np.full((self.num_envs, 1, 1), OPEN)
        
        # Concatenate to get (num_envs, 1, 8)
        trajectories = np.concatenate([trajectories, gripper_state], axis=-1)
        
        # Execute trajectory
        self.env.robot_controller.execute_trajectory(trajectories)
        
        # All should succeed since it's just gripper control
        success = np.ones(self.num_envs, dtype=bool)
        
        return trajectories, success

    def CloseGripper(
        self
        ):
        """Close the gripper in every sim slot without moving the arm.

        Returns:
            (num_envs, 1, 8) one-step trajectories and an all-True (num_envs,) success array.
        """
        # Get current joint state (num_envs, 7)
        current_joint_state = self.env.get_joint_state().to("cuda:0")
        
        # Create single-step trajectory (num_envs, 1, 7)
        trajectories = current_joint_state.unsqueeze(1).cpu().detach().numpy()
        
        # Add CLOSED gripper state (num_envs, 1, 1)
        gripper_state = np.full((self.num_envs, 1, 1), CLOSED)
        
        # Concatenate to get (num_envs, 1, 8)
        trajectories = np.concatenate([trajectories, gripper_state], axis=-1)
        
        # Execute trajectory
        self.env.robot_controller.execute_trajectory(trajectories)
        
        # All should succeed since it's just gripper control
        success = np.ones(self.num_envs, dtype=bool)
        
        return trajectories, success
    
    def Move(
        self,
        goal_poses: torch.Tensor       # in sapien frame, shape: (batch_size, 7)
        ):
        """Plan and execute end-effector moves to goal poses, resetting slots that fail.

        Args:
            goal_poses: (B, 7) end-effector poses in the sim frame, B <= num_envs.

        Returns:
            (B, T, 8) joint + gripper trajectories and (B,) success (planned and no object moved).
        """
        # Get batch size from input
        batch_size = goal_poses.shape[0]
        
        # Assert that batch_size is not greater than num_envs
        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"

        initial_state = self.env.get_state().clone()
        initial_object_poses = self.env.object_poses_tensor.clone()  # (num_envs, num_objects, 7)

        self.update_collision_world()

        curobo_goal_poses = goal_poses.clone()
        curobo_goal_poses[..., 0] = curobo_goal_poses[..., 0] + 0.615

        # Get joint state torch tensor (num_envs, 7) - only use first batch_size
        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]
        # Get gripper state numpy array (num_envs, 1) - only use first batch_size
        current_gripper_state = self.env.robot_controller.gripper_state_batch[:batch_size].copy()   # This is a numpy array, and should remain a numpy array, because trajectory execution happens with numpy
        
        trajectories, success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_goal_poses,
            disable_collision_links=[]
        )

        trajectories = trajectories.cpu().detach().numpy()
        success = success.cpu().detach().numpy()
        current_gripper_state = np.repeat(current_gripper_state[:, np.newaxis, :], trajectories.shape[1], axis=1)

        trajectories = np.concatenate(
            [trajectories, current_gripper_state],
            axis=-1
        )

        # If nothing was successful:
        if np.all(success == False):
            return trajectories, success

        self.env.robot_controller.execute_trajectory(trajectories)

        # Check if any objects have moved beyond threshold (checking full pose: position + rotation)
        current_object_poses = self.env.object_poses_tensor  # (num_envs, num_objects, 7)
        
        # Compute pose distances for all objects (position + rotation)
        pose_distances = self.compute_pose_distance(initial_object_poses, current_object_poses)  # (num_envs, num_objects)
        
        # Check if any object has moved beyond threshold - only check first batch_size
        max_movement = pose_distances[:batch_size].max(dim=1)[0]  # (batch_size,) - max movement per environment
        no_unwanted_movement = max_movement < self.movement_threshold  # (batch_size,)
        no_unwanted_movement = no_unwanted_movement.cpu().numpy()
        
        # Combine planning success with movement check
        success = success & no_unwanted_movement

        # Reset failed environments to initial state (only reset first batch_size)
        current_state = self.env.get_state().clone()
        failures = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda:0")
        failures[:batch_size] = torch.tensor(~success, device="cuda:0", dtype=torch.bool)
        current_state[failures] = initial_state[failures]

        self.env.set_state(current_state)
        if self.env.render_mode is not None:
            self.env.render()

        return trajectories, success

    def Pick(
        self,
        grasp_parameters: torch.Tensor,       # in sapien frame, shape: (batch_size, 9), with (x, y, z, qw, qx, qy, qz, grasp_cost, grasp_object_id)
        ) -> tuple[np.ndarray, np.ndarray]:
        """Plan, execute, and judge a batch of grasp-and-lift actions.

        Args:
            grasp_parameters: (B, 9) rows of (x, y, z, qw, qx, qy, qz, grasp_cost, object_index), sim frame.

        Returns:
            Same 5-tuple as judge_pick.
        """
        plan = self.plan_pick(grasp_parameters)
        if "result" in plan:
            return plan["result"]
        self.env.robot_controller.execute_trajectory(plan["trajectories"])
        return self.judge_pick(plan)

    def plan_pick(self, grasp_parameters: torch.Tensor) -> dict:
        """Plan approach, grasp, and lift motions for a batch of grasps, without executing them.

        Args:
            grasp_parameters: (B, 9) rows of (x, y, z, qw, qx, qy, qz, grasp_cost, object_index), sim frame.

        Returns:
            Plan dict for judge_pick; its "result" holds the final outcome when no row planned.
        """
        # Get batch size from input
        batch_size = grasp_parameters.shape[0]
        grasp_poses = grasp_parameters[..., :7].clone()
        object_indices = [int(idx.item()) for idx in grasp_parameters[..., -1]]

        # Assert that batch_size is not greater than num_envs
        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"
        assert len(object_indices) == batch_size, f"object_indices length ({len(object_indices)}) must match batch_size ({batch_size})"

        lift_distances=torch.tensor([POST_GRASP_LIFT for _ in range(batch_size)], device="cuda:0", dtype=torch.float32)

        initial_state = self.env.get_state().clone()
        initial_object_poses = self.env.object_poses_tensor.clone()  # (num_envs, num_objects, 7)
        rewards_before_execution = self.env.compute_dense_reward()[:batch_size]
        
        self.update_collision_world()

        curobo_grasp_poses = grasp_poses.clone()
        curobo_grasp_poses[..., 0] = curobo_grasp_poses[..., 0] + 0.615

        curobo_pre_grasp_poses = curobo_grasp_poses.clone()
        curobo_pre_grasp_poses = move_pose_along_local_z(curobo_pre_grasp_poses, -GRASP_DEPTH)
        curobo_pre_grasp_poses = torch.tensor(curobo_pre_grasp_poses, dtype=curobo_grasp_poses.dtype, device=curobo_grasp_poses.device)

        curobo_lift_poses = curobo_grasp_poses.clone()
        curobo_lift_poses[..., 2] = curobo_lift_poses[..., 2] + lift_distances

        # Get joint state torch tensor (num_envs, 7) - only use first batch_size
        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]

        pre_grasp_trajectories, pre_grasp_success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_pre_grasp_poses,
            disable_collision_links=[]
        )
        
        grasp_trajectories, grasp_success = self.plan_to_goal_poses(
            current_joints=pre_grasp_trajectories[:, -1],
            goal_poses=curobo_grasp_poses,
            plan_config=self.along_z_plan_config,
            disable_collision_links=[]
        )

        lift_trajectories, lift_success = self.plan_to_goal_poses(
            current_joints=grasp_trajectories[:, -1],
            goal_poses=curobo_lift_poses,
            plan_config=self.lift_plan_config,
            disable_collision_links=[]
        )

        # Convert to numpy
        pre_grasp_trajectories = pre_grasp_trajectories.cpu().detach().numpy()
        grasp_trajectories = grasp_trajectories.cpu().detach().numpy()
        lift_trajectories = lift_trajectories.cpu().detach().numpy()
        pre_grasp_success = pre_grasp_success.cpu().detach().numpy()
        grasp_success = grasp_success.cpu().detach().numpy()
        lift_success = lift_success.cpu().detach().numpy()
        
        # Combine success: pre_grasp, grasp, and lift must all succeed
        success = pre_grasp_success & grasp_success & lift_success

        # Classify failure types per environment
        failure_types = [None] * batch_size
        for idx in range(batch_size):
            if not pre_grasp_success[idx]:
                failure_types[idx] = "pre_grasp_motion_plan"
            elif not grasp_success[idx]:
                failure_types[idx] = "grasp_motion_plan"
            elif not lift_success[idx]:
                failure_types[idx] = "lift_motion_plan"

        # Form the gripper state (using batch_size)
        pre_grasp_gripper = np.full((batch_size, pre_grasp_trajectories.shape[1]), OPEN)
        grasp_gripper = np.full((batch_size, grasp_trajectories.shape[1]), OPEN)
        grasp_gripper[:, -1] = CLOSED  # Last element is where the grasp should happen
        lift_gripper = np.full((batch_size, lift_trajectories.shape[1]), CLOSED)

        # Stack trajectories and gripper states
        trajectories = np.concatenate([pre_grasp_trajectories, grasp_trajectories, lift_trajectories], axis=1)  # (batch_size, pre_grasp_len + grasp_len + lift_len, 7)
        gripper_states = np.concatenate([pre_grasp_gripper, grasp_gripper, lift_gripper], axis=1)  # (batch_size, pre_grasp_len + grasp_len + lift_len)
        gripper_states = gripper_states[:, :, np.newaxis]  # (batch_size, total_len, 1)

        # Concatenate trajectories with gripper states
        trajectories = np.concatenate([trajectories, gripper_states], axis=-1)  # (batch_size, total_len, 8)

        plan = dict(grasp_parameters=grasp_parameters, object_indices=object_indices, lift_distances=lift_distances,
                    initial_state=initial_state, initial_object_poses=initial_object_poses,
                    rewards_before_execution=rewards_before_execution, success=success,
                    failure_types=failure_types, trajectories=trajectories, settle_steps=0)
        # If nothing was successful (all motion plans failed):
        if np.all(success == False):
            # State is unchanged (no execution happened), use initial_state for all, no trajectory
            plan["result"] = [], [], [], [], [(grasp_parameters[i], failure_types[i], initial_state[i], None)
                                              for i in range(batch_size)]
        return plan

    def judge_pick(self, plan: dict):
        """Check which executed pick plans lifted their object by at least half the lift height.

        Reads the end states from the sim's first B slots.

        Args:
            plan: Dict from plan_pick.

        Returns:
            States, trajectories, actions and rewards of the successful rows, and
            (action, failure_type, state, trajectory or None) tuples for the failed rows.
        """
        grasp_parameters, object_indices, lift_distances = (
            plan["grasp_parameters"], plan["object_indices"], plan["lift_distances"])
        initial_state, initial_object_poses = plan["initial_state"], plan["initial_object_poses"]
        rewards_before_execution, success = plan["rewards_before_execution"], plan["success"]
        failure_types, trajectories = list(plan["failure_types"]), plan["trajectories"]
        batch_size = grasp_parameters.shape[0]

        # Check if objects are grasped for each environment (only check first batch_size)
        # Object is grasped if its z position is at least lift_distance/2 above initial z position
        current_object_poses = self.env.object_poses_tensor  # (num_envs, num_objects, 7)

        is_grasped = np.zeros(batch_size, dtype=bool)
        for env_idx in range(batch_size):
            object_index = object_indices[env_idx]
            initial_z = initial_object_poses[env_idx, object_index, 2].item()  # Initial z position
            current_z = current_object_poses[env_idx, object_index, 2].item()  # Current z position
            lift_threshold = initial_z + (lift_distances[env_idx].item() / 2)
            is_grasped[env_idx] = current_z >= lift_threshold

        # Update failure types for grasp execution failures
        for idx in range(batch_size):
            if success[idx] and not is_grasped[idx]:
                failure_types[idx] = "grasp_execution"

        success = success & is_grasped #& no_unwanted_movement

        # Capture post-execution states before resetting failed envs
        current_state = self.env.get_state().clone()[:batch_size]
        rewards_after_execution = self.env.compute_dense_reward()[:batch_size]

        rewards = rewards_after_execution - rewards_before_execution    # Volume of objects transferred to the shelf

        # Collect failed actions info with their states and trajectories
        # Motion plan failures: initial_state, no trajectory
        # Execution failures: post-execution state, with trajectory
        failed_actions_info = []
        for i in range(batch_size):
            if not success[i]:
                if failure_types[i] in ("pre_grasp_motion_plan", "grasp_motion_plan", "lift_motion_plan"):
                    failed_actions_info.append((grasp_parameters[i], failure_types[i], initial_state[i], None))
                else:
                    failed_actions_info.append((grasp_parameters[i], failure_types[i], current_state[i], trajectories[i].copy()))

        if success.sum() == 0:
            # Everything failed, return empty lists
            return [], [], [], [], failed_actions_info

        successful_states = current_state[success]
        successful_trajectories = trajectories[success]
        successful_actions = grasp_parameters[success]
        rewards = rewards[success]

        return successful_states, successful_trajectories, successful_actions, rewards, failed_actions_info
        
    def Place(
        self,
        place_parameters: torch.Tensor,       # in sapien frame, shape: (batch_size, 8), with (x, y, z, qw, qx, qy, qz, object_index)
        ):
        """Plan, execute, and judge a batch of shelf placements of the held object.

        Args:
            place_parameters: (B, 8) rows of (x, y, z, qw, qx, qy, qz, object_index), sim frame; one object for all rows.

        Returns:
            Same 5-tuple as judge_pick.
        """
        plan = self.plan_place(place_parameters)
        if "result" in plan:
            return plan["result"]
        self.env.robot_controller.execute_trajectory(plan["trajectories"])
        return self.judge_place(plan)

    def plan_place(self, place_parameters: torch.Tensor) -> dict:
        """Plan carrying the held object to each shelf pose, releasing it, and retracing, without executing.

        Args:
            place_parameters: (B, 8) rows of (x, y, z, qw, qx, qy, qz, object_index), sim frame; one object for all rows.

        Returns:
            Plan dict for judge_place; its "result" holds the final outcome when no row planned.
        """
        # Get batch size from input
        batch_size = place_parameters.shape[0]
        
        # Extract place_poses and object_index from the tensor
        place_poses = place_parameters[..., :7].clone()
        object_indices = [int(round(idx.item())) for idx in place_parameters[..., -1]]
        
        # Assert that batch_size is not greater than num_envs
        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"
        assert len(object_indices) == batch_size, f"object_indices length ({len(object_indices)}) must match batch_size ({batch_size})"
        assert all(idx == object_indices[0] for idx in object_indices), "All placements in a batch must be for the same object"

        initial_state = self.env.get_state().clone()
        initial_object_poses = self.env.object_poses_tensor.clone()  # (num_envs, num_objects, 7)
        rewards_before_execution = self.env.compute_dense_reward()[:batch_size]
        
        object_index = object_indices[0]  # All should be the same
        object_name = self.env.object_names[object_index]
        
        self.update_collision_world()
        self.attach_object_to_robot(object_name)

        # Find objects already in shelf and disable them during planning
        _, object_indices_in_shelf = self.env.number_of_objects_in_shelf(env_idx=0)
        objects_in_shelf = [self.env.object_names[i] for i in object_indices_in_shelf]

        curobo_place_poses = place_poses.clone()
        curobo_place_poses[..., 0] = curobo_place_poses[..., 0] + 0.615

        # Get joint state torch tensor (num_envs, 7) - only use first batch_size
        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]
        
        # print(f"Objects in shelf: {objects_in_shelf}")
        place_trajectories, place_success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_place_poses,
            plan_config=self.placement_plan_config,
            objects_to_ignore=objects_in_shelf,
            disable_collision_links=[]
        )
        self.detach_object_from_robot(object_name)

        place_trajectories = place_trajectories.cpu().detach().numpy()
        place_success = place_success.cpu().detach().numpy()

        # Form the gripper state (all CLOSED for forward place) - using batch_size
        place_gripper = np.full((batch_size, place_trajectories.shape[1]), CLOSED)
        place_gripper = place_gripper[:, :, np.newaxis]  # (batch_size, traj_len, 1)

        # Concatenate forward trajectories with gripper states
        place_trajectories_forward = np.concatenate([place_trajectories, place_gripper], axis=-1)  # (batch_size, traj_len, 8)

        # Reverse placement trajectories to go back (with OPEN gripper)
        place_trajectories_reversed = np.flip(place_trajectories, axis=1)  # Reverse along time axis
        place_gripper_open = np.full((batch_size, place_trajectories_reversed.shape[1]), OPEN)
        place_gripper_open = place_gripper_open[:, :, np.newaxis]  # (batch_size, traj_len, 1)
        place_trajectories_reversed = np.concatenate([place_trajectories_reversed, place_gripper_open], axis=-1)  # (batch_size, traj_len, 8)

        # Stack forward and reversed trajectories
        place_trajectories = np.concatenate([place_trajectories_forward, place_trajectories_reversed], axis=1)

        # Classify failure types per environment
        failure_types = [None] * batch_size
        for idx in range(batch_size):
            if not place_success[idx]:
                failure_types[idx] = "place_motion_plan"

        plan = dict(place_parameters=place_parameters, object_name=object_name, initial_state=initial_state,
                    rewards_before_execution=rewards_before_execution, success=place_success,
                    failure_types=failure_types, trajectories=place_trajectories, settle_steps=0)
        # If nothing was successful (all motion plans failed):
        if np.all(place_success == False):
            plan["result"] = [], [], [], [], [(place_parameters[i], failure_types[i], initial_state[i], None)
                                              for i in range(batch_size)]
        return plan

    def judge_place(self, plan: dict):
        """Check which executed shelf placements left the object inside the shelf.

        Reads the end states from the sim's first B slots.

        Args:
            plan: Dict from plan_place.

        Returns:
            Same 5-tuple as judge_pick.
        """
        place_parameters, object_name, initial_state = (
            plan["place_parameters"], plan["object_name"], plan["initial_state"])
        rewards_before_execution, place_success = plan["rewards_before_execution"], plan["success"]
        failure_types, place_trajectories = list(plan["failure_types"]), plan["trajectories"]
        batch_size = place_parameters.shape[0]

        is_placed = np.zeros(batch_size, dtype=bool)
        for env_idx in range(batch_size):
            is_placed[env_idx] = self.env.check_object_inside_shelf(object_name, env_idx=env_idx)

        # Update failure types for placement execution failures
        for idx in range(batch_size):
            if place_success[idx] and not is_placed[idx]:
                failure_types[idx] = "place_execution"

        place_success = place_success & is_placed

        # Capture post-execution states before resetting failed envs
        current_state = self.env.get_state().clone()[:batch_size]
        rewards_after_execution = self.env.compute_dense_reward()[:batch_size]
        rewards = rewards_after_execution - rewards_before_execution    # Volume of objects transferred to the shelf

        # Collect failed actions info with their states and trajectories
        failed_actions_info = []
        for i in range(batch_size):
            if not place_success[i]:
                if failure_types[i] == "place_motion_plan":
                    failed_actions_info.append((place_parameters[i], failure_types[i], initial_state[i], None))
                else:
                    failed_actions_info.append((place_parameters[i], failure_types[i], current_state[i], place_trajectories[i].copy()))

        if place_success.sum() == 0:
            # Everything failed, return empty lists
            return [], [], [], [], failed_actions_info

        successful_states = current_state[place_success]
        successful_trajectories = place_trajectories[place_success]
        successful_actions = place_parameters[place_success]
        rewards = rewards[place_success]

        return successful_states, successful_trajectories, successful_actions, rewards, failed_actions_info

    def PlaceStacking(
        self,
        place_parameters: torch.Tensor,       # (batch_size, 8): (x, y, z, qw, qx, qy, qz, object_index)
        base_idx,                             # int, or per-row sequence of ints (-1 = no base)
        pre_base_pose: torch.Tensor = None,   # (7,) or (batch_size, 7); None = read from env per row
        chain_below: list = None,             # tower objects below the target (accepted for API compat; kept as obstacles)
        trans_tol: float = 0.01,
        quat_tol: float = 0.05,
        free_yaw: bool = False,               # allow any rotation about world z at the place pose
        descend: float = 0.0,                 # after reaching the place pose, lower this far (m) before release
        lift_fallback: bool = False,          # retry a failed post-release lift (supports ignored, then back-out)
        ):
        """Plan, execute, settle, and judge a batch of stacking placements of the held object.

        Args:
            place_parameters: (B, 8) rows of (x, y, z, qw, qx, qy, qz, object_index), sim frame; one object for all rows.
            base_idx: Index of the object to stack on, or one per row (-1 = no base).
            pre_base_pose: (7,) or (B, 7) base pose before the action; None reads it from the sim.
            chain_below: Unused; kept for API compatibility.
            trans_tol: Largest base shift (m) that does not count as knocking it.
            quat_tol: Largest base rotation change (1 - |q . q0|) that does not count as knocking it.
            free_yaw: Allow any rotation about world z at the place pose.
            descend: Distance (m) to lower the object below the place pose before release.
            lift_fallback: Retry a failed post-release lift with supports ignored, then by backing out.

        Returns:
            Same 5-tuple as judge_pick.
        """
        plan = self.plan_place_stacking(place_parameters, base_idx, pre_base_pose, chain_below, trans_tol,
                                        quat_tol, free_yaw, descend, lift_fallback)
        if "result" in plan:
            return plan["result"]
        self.env.robot_controller.execute_trajectory(plan["trajectories"])
        # Let the dropped object settle on the tower.
        self.env.wait_for_stability(max_steps=plan["settle_steps"])
        return self.judge_place_stacking(plan)

    def plan_place_stacking(self, place_parameters, base_idx, pre_base_pose=None, chain_below=None,
                            trans_tol=0.01, quat_tol=0.05, free_yaw=False, descend=0.0, lift_fallback=False):
        """Plan carrying the held object onto its base, releasing it, and lifting away, without executing.

        Tower objects stay as collision obstacles; only the held object is ignored.

        Args:
            place_parameters: (B, 8) rows of (x, y, z, qw, qx, qy, qz, object_index), sim frame; one object for all rows.
            base_idx: Index of the object to stack on, or one per row (-1 = no base, fails as "no_base").
            pre_base_pose: (7,) or (B, 7) base pose before the action; None reads it from the sim.
            chain_below: Unused; kept for API compatibility.
            trans_tol: Largest base shift (m) that does not count as knocking it.
            quat_tol: Largest base rotation change (1 - |q . q0|) that does not count as knocking it.
            free_yaw: Allow any rotation about world z at the place pose.
            descend: Distance (m) to lower the object below the place pose before release.
            lift_fallback: Retry a failed post-release lift with supports ignored, then by backing out.

        Returns:
            Plan dict for judge_place_stacking; its "result" holds the final outcome when no row planned.
        """
        batch_size = place_parameters.shape[0]
        place_poses = place_parameters[..., :7].clone()
        object_indices = [int(round(idx.item())) for idx in place_parameters[..., -1]]

        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"
        assert len(object_indices) == batch_size
        assert all(idx == object_indices[0] for idx in object_indices), \
            "All placements in a batch must be for the same object"

        initial_state = self.env.get_state().clone()
        initial_object_poses = self.env.object_poses_tensor.clone()
        rewards_before_execution = self.env.compute_dense_reward()[:batch_size]

        object_index = object_indices[0]
        object_name = self.env.object_names[object_index]

        if isinstance(base_idx, (int, np.integer)):
            bases = [int(base_idx)] * batch_size
        else:
            bases = [int(b) for b in base_idx]
        assert len(bases) == batch_size, f"base_idx length {len(bases)} != batch_size {batch_size}"
        if pre_base_pose is None:
            pre_base_poses = [
                initial_object_poses[i, b] if b >= 0 else torch.zeros(7, device=initial_object_poses.device)
                for i, b in enumerate(bases)
            ]
        elif pre_base_pose.ndim == 1:
            pre_base_poses = [pre_base_pose] * batch_size
        else:
            pre_base_poses = [pre_base_pose[i] for i in range(batch_size)]

        self.update_collision_world()
        self.attach_object_to_robot(object_name)

        # Ignore ONLY the moved object in the world collision: it is attached to
        # the gripper, so its stale world-frame copy would otherwise collide with
        # the attached body. The tower objects (chain_below) MUST stay as
        # obstacles — disabling them lets cuRobo plan straight through the stack
        # and knock it over.
        objects_to_ignore = [object_name]

        curobo_place_poses = place_poses.clone()
        curobo_place_poses[..., 0] = curobo_place_poses[..., 0] + 0.615

        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]

        # Full 6-DOF pose enforced for placement (no pose_cost_metric): the
        # object must land at exactly prev_action.place_pose ⊗ grasp. The
        # shelf Place freed pitch (placement_plan_config) for easier shelf
        # reach, but that orientation slack topples small stacking towers.
        place_trajectories, place_success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_place_poses,
            objects_to_ignore=objects_to_ignore,
            disable_collision_links=[],
            **({"plan_config": self.stack_place_plan_config} if free_yaw else {}),
        )
        if descend > 0:
            # Lower the held object straight down by `descend` (orientation and world x, y held)
            # into contact with its support before releasing, instead of dropping it from the
            # place pose. The supports are ignored for this short final segment so the goal can
            # touch them.
            curobo_down_poses = curobo_place_poses.clone()
            curobo_down_poses[..., 2] = curobo_down_poses[..., 2] - descend
            supports = sorted({self.env.object_names[b] for b in bases if b >= 0})
            down_trajectories, down_success = self.plan_to_goal_poses(
                current_joints=place_trajectories[:, -1],
                goal_poses=curobo_down_poses,
                plan_config=self.lift_plan_config,
                objects_to_ignore=objects_to_ignore + supports,
                disable_collision_links=[],
            )
            place_trajectories = torch.cat([place_trajectories, down_trajectories], dim=1)
            place_success = place_success & down_success
        self.detach_object_from_robot(object_name)

        # If no row could plan the placement, skip the lift entirely (nothing to
        # retreat from) and report place_motion_plan for the whole batch.
        if not bool(place_success.any()):
            return {"result": ([], [], [], [], [(place_parameters[i], "place_motion_plan", initial_state[i], None)
                                                for i in range(batch_size)])}

        # After releasing, lift the (now empty) gripper straight up before the
        # next grasp. Same constraint as Pick's lift: `lift_plan_config` uses a
        # PoseCostMetric (hold_vec_weight=[1,1,1,1,1,0], world frame) that holds
        # orientation + world x,y and frees ONLY world z, so cuRobo can only
        # translate vertically — the gripper retreats straight off the tower
        # instead of sweeping sideways through it. Plan from the place end
        # config; tower stays as an obstacle (only the moved object is ignored).
        curobo_lift_poses = curobo_place_poses.clone()
        curobo_lift_poses[..., 2] = curobo_lift_poses[..., 2] + POST_GRASP_LIFT
        lift_trajectories, lift_success = self.plan_to_goal_poses(
            current_joints=place_trajectories[:, -1],
            goal_poses=curobo_lift_poses,
            plan_config=self.lift_plan_config,
            objects_to_ignore=objects_to_ignore,
            disable_collision_links=[],
        )
        if lift_fallback and bool((place_success & ~lift_success).any()):
            # The release config often touches the support (fingers reach below the held object
            # after `descend`), so the straight-up lift starts in collision. Fallback 1: the same
            # vertical lift with the supports ignored (moving up cannot hit what is below).
            supports = sorted({self.env.object_names[b] for b in bases if b >= 0})
            fb_traj, fb_ok = self.plan_to_goal_poses(
                current_joints=place_trajectories[:, -1],
                goal_poses=curobo_lift_poses,
                plan_config=self.lift_plan_config,
                objects_to_ignore=objects_to_ignore + supports,
                disable_collision_links=[],
            )
            lift_trajectories, lift_success = _merge_rows(lift_trajectories, lift_success, fb_traj, fb_ok)
        if lift_fallback and bool((place_success & ~lift_success).any()):
            # Fallback 2: back out along the gripper's approach axis (EE -z), free-form plan.
            back = curobo_place_poses.clone()
            if descend > 0:
                back[..., 2] = back[..., 2] - descend
            approach = quaternion_to_matrix(back[:, 3:7].cpu().numpy(), format="wxyz")[:, :, 2]
            back[:, :3] = back[:, :3] - POST_GRASP_LIFT * torch.as_tensor(approach, dtype=back.dtype,
                                                                           device=back.device)
            supports = sorted({self.env.object_names[b] for b in bases if b >= 0})
            fb_traj, fb_ok = self.plan_to_goal_poses(
                current_joints=place_trajectories[:, -1],
                goal_poses=back,
                objects_to_ignore=objects_to_ignore + supports,
                disable_collision_links=[],
            )
            lift_trajectories, lift_success = _merge_rows(lift_trajectories, lift_success, fb_traj, fb_ok)

        place_trajectories = place_trajectories.cpu().detach().numpy()
        place_success = place_success.cpu().detach().numpy()
        lift_trajectories = lift_trajectories.cpu().detach().numpy()
        lift_success = lift_success.cpu().detach().numpy()

        # Forward place (gripper CLOSED) then straight-up lift (gripper OPEN).
        # The CLOSED->OPEN transition at the place pose is the release.
        place_gripper = np.full((batch_size, place_trajectories.shape[1]), CLOSED)[:, :, np.newaxis]
        place_forward = np.concatenate([place_trajectories, place_gripper], axis=-1)
        lift_gripper = np.full((batch_size, lift_trajectories.shape[1]), OPEN)[:, :, np.newaxis]
        lift_forward = np.concatenate([lift_trajectories, lift_gripper], axis=-1)
        place_trajectories = np.concatenate([place_forward, lift_forward], axis=1)

        # Place must succeed; a failed lift is its own failure type.
        failure_types = [None] * batch_size
        for idx in range(batch_size):
            if not place_success[idx]:
                failure_types[idx] = "place_motion_plan"
            elif not lift_success[idx]:
                failure_types[idx] = "lift_motion_plan"

        place_success = place_success & lift_success
        for idx in range(batch_size):
            if bases[idx] < 0:
                place_success[idx] = False
                failure_types[idx] = "no_base"

        plan = dict(place_parameters=place_parameters, object_index=object_index, bases=bases,
                    pre_base_poses=pre_base_poses, trans_tol=trans_tol, quat_tol=quat_tol,
                    initial_state=initial_state, rewards_before_execution=rewards_before_execution,
                    success=place_success, failure_types=failure_types, trajectories=place_trajectories,
                    settle_steps=200)
        if np.all(place_success == False):
            plan["result"] = [], [], [], [], [(place_parameters[i], failure_types[i], initial_state[i], None)
                                              for i in range(batch_size)]
        return plan

    def judge_place_stacking(self, plan: dict):
        """Check which executed stacking placements rest on their base without knocking it.

        Reads the settled end states from the sim's first B slots.

        Args:
            plan: Dict from plan_place_stacking.

        Returns:
            Same 5-tuple as judge_pick.
        """
        place_parameters, object_index, bases = plan["place_parameters"], plan["object_index"], plan["bases"]
        pre_base_poses, trans_tol, quat_tol = plan["pre_base_poses"], plan["trans_tol"], plan["quat_tol"]
        initial_state, rewards_before_execution = plan["initial_state"], plan["rewards_before_execution"]
        place_success, failure_types = plan["success"].copy(), list(plan["failure_types"])
        place_trajectories = plan["trajectories"]
        batch_size = place_parameters.shape[0]

        is_placed = np.zeros(batch_size, dtype=bool)
        S = self.env.stacked_matrix()
        for env_idx in range(batch_size):
            if not place_success[env_idx]:
                continue
            # Stacking success: moved is on base AND base wasn't knocked.
            b = bases[env_idx]
            stacked = bool(S[env_idx, object_index, b])
            cur_base_pose = self.env.get_object_pose(self.env.object_names[b])[env_idx]
            pre = pre_base_poses[env_idx].to(cur_base_pose.device).to(cur_base_pose.dtype)
            trans_drift = torch.norm(cur_base_pose[:3] - pre[:3]).item()
            quat_dot = torch.dot(cur_base_pose[3:], pre[3:]).item()
            quat_drift = 1.0 - abs(quat_dot)
            base_ok = trans_drift <= trans_tol and quat_drift <= quat_tol
            is_placed[env_idx] = stacked and base_ok

        for idx in range(batch_size):
            if place_success[idx] and not is_placed[idx]:
                # Distinguish stack-on-failure vs base-knock for analysis.
                if bool(S[idx, object_index, bases[idx]]):
                    failure_types[idx] = "knocked_base"
                else:
                    failure_types[idx] = "place_execution"

        place_success = place_success & is_placed

        current_state = self.env.get_state().clone()[:batch_size]
        rewards_after_execution = self.env.compute_dense_reward()[:batch_size]
        rewards = rewards_after_execution - rewards_before_execution

        failed_actions_info = []
        for i in range(batch_size):
            if not place_success[i]:
                if failure_types[i] in ("place_motion_plan", "no_base"):
                    failed_actions_info.append((place_parameters[i], failure_types[i], initial_state[i], None))
                else:
                    failed_actions_info.append((place_parameters[i], failure_types[i], current_state[i], place_trajectories[i].copy()))

        if place_success.sum() == 0:
            return [], [], [], [], failed_actions_info

        successful_states = current_state[place_success]
        successful_trajectories = place_trajectories[place_success]
        successful_actions = place_parameters[place_success]
        rewards = rewards[place_success]

        return successful_states, successful_trajectories, successful_actions, rewards, failed_actions_info

    def Push(
        self,
        push_start: torch.Tensor,
        push_direction: torch.Tensor,
        push_distance: torch.Tensor,
        ):
        """Plan and execute pushes along an axis of the start pose, resetting slots that fail.

        Args:
            push_start: (B, 7) gripper start poses in the sim frame.
            push_direction: (B,) axis of the start pose to push along (0 = x, 1 = y, 2 = z).
            push_distance: (B,) signed push distance (m) along that axis.

        Returns:
            (B, T, 8) approach, push, and retreat trajectories and (B,) planning success.
        """
        # Get batch size from input
        batch_size = push_start.shape[0]
        
        # Assertions for input sizes
        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"
        assert push_start.shape == (batch_size, 7), f"push_start must be (batch_size={batch_size}, 7), got {push_start.shape}"
        assert push_direction.shape == (batch_size,), f"push_direction must be (batch_size={batch_size},), got {push_direction.shape}"
        assert push_distance.shape == (batch_size,), f"push_distance must be (batch_size={batch_size},), got {push_distance.shape}"
    
        initial_state = self.env.get_state().clone()
        
        self.update_collision_world()

        # Compute push end poses in parallel
        # Extract position and quaternion from push_start
        push_start_np = push_start.cpu().detach().numpy()
        pos = push_start_np[:, :3]  # (batch_size, 3)
        quat = push_start_np[:, 3:7]  # (batch_size, 4) in wxyz format
        
        # Get rotation matrices: (batch_size, 3, 3)
        R = quaternion_to_matrix(quat, format='wxyz')
        
        # Gather the appropriate axis for each environment based on push_direction
        # push_direction: (batch_size,) with values 0, 1, or 2 for X, Y, Z
        axis_indices = push_direction.cpu().numpy().astype(int)  # (batch_size,)
        push_distance_np = push_distance.cpu().numpy()  # (batch_size,)
        
        # For each env, select the appropriate column of R
        # R[:, :, axis_indices[i]] gives the local axis direction
        env_indices = np.arange(batch_size)
        local_axes = R[env_indices, :, axis_indices]  # (batch_size, 3)
        
        # Compute new positions: pos + push_distance * local_axis
        pos_new = pos + (push_distance_np[:, np.newaxis] * local_axes)
        
        # Construct push_end poses (orientation stays the same)
        push_end_np = push_start_np.copy()
        push_end_np[:, :3] = pos_new
        push_end = torch.tensor(push_end_np, dtype=push_start.dtype, device=push_start.device)

        # Convert to cuRobo frame
        curobo_push_start = push_start.clone()
        curobo_push_start[..., 0] = curobo_push_start[..., 0] + 0.615        
        curobo_push_end = push_end.clone()
        curobo_push_end[..., 0] = curobo_push_end[..., 0] + 0.615

        # Get joint state torch tensor (num_envs, 7) - only use first batch_size
        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]
        
        # ========== Step 1: Plan to push_start without disabling anything ==========
        approach_trajectories, approach_success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_push_start,
            disable_collision_links=[]
        )
        
        # ========== Step 2: Plan push movement for each axis separately ==========
        # Start joints for push are the end of approach trajectories
        push_start_joints = approach_trajectories[:, -1, :]  # (batch_size, 7)
        
        # Initialize push trajectories and success
        traj_len = self.motion_gen.trajopt_solver.traj_tsteps
        push_trajectories = push_start_joints.unsqueeze(1).repeat(1, traj_len, 1)  # Default: stay at start
        push_success = torch.zeros(batch_size, dtype=torch.bool, device="cuda:0")
        
        # Map axis index to plan config
        axis_plan_configs = {
            0: self.along_x_plan_config,
            1: self.along_y_plan_config,
            2: self.along_z_plan_config,
        }
        
        # Collision links to disable for push movement (panda hand to attached object)
        push_disable_links = [
            "panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object"
            ]
        
        # Group environments by axis and plan for each axis that has environments
        for axis_idx in [0, 1, 2]:
            # Find environments using this axis
            axis_mask = (axis_indices == axis_idx)
            if not np.any(axis_mask):
                continue  # Skip if no environments use this axis
            
            env_indices_for_axis = torch.tensor(np.where(axis_mask)[0], device="cuda:0", dtype=torch.long)
            
            # Get the start joints and goal poses for these environments
            axis_start_joints = push_start_joints[env_indices_for_axis]
            axis_goal_poses = curobo_push_end[env_indices_for_axis]
            
            # Plan for this axis
            axis_trajectories, axis_success = self.plan_to_goal_poses(
                current_joints=axis_start_joints,
                goal_poses=axis_goal_poses,
                disable_collision_links=push_disable_links,
                plan_config=axis_plan_configs[axis_idx]
            )
            
            # Store results back into full arrays
            push_trajectories[env_indices_for_axis] = axis_trajectories
            push_success[env_indices_for_axis] = axis_success
        
        # Combine approach and push success
        success = approach_success & push_success
        
        # ========== Combine trajectories and add gripper states ==========
        approach_trajectories = approach_trajectories.cpu().detach().numpy()
        push_trajectories = push_trajectories.cpu().detach().numpy()
        success = success.cpu().detach().numpy()
        
        # Gripper stays closed throughout - using batch_size
        approach_gripper = np.full((batch_size, approach_trajectories.shape[1], 1), CLOSED)
        push_gripper = np.full((batch_size, push_trajectories.shape[1], 1), CLOSED)
        
        # Concatenate gripper states
        approach_trajectories = np.concatenate([approach_trajectories, approach_gripper], axis=-1)
        push_trajectories = np.concatenate([push_trajectories, push_gripper], axis=-1)
        
        # Reverse push trajectories to go back
        push_trajectories_reversed = np.flip(push_trajectories, axis=1)  # Reverse along time axis
        
        # Stack approach, push (forward), and push (backward) trajectories
        trajectories = np.concatenate([approach_trajectories, push_trajectories, push_trajectories_reversed], axis=1)
        
        # If nothing was successful, return early
        if np.all(success == False):
            return trajectories, success
        
        # Execute trajectory
        self.env.robot_controller.execute_trajectory(trajectories)
        
        # Reset failed environments to initial state (only reset first batch_size)
        current_state = self.env.get_state().clone()
        failures = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda:0")
        failures[:batch_size] = torch.tensor(~success, device="cuda:0", dtype=torch.bool)
        current_state[failures] = initial_state[failures]
        
        self.env.set_state(current_state)
        if self.env.render_mode is not None:
            self.env.render()
        
        return trajectories, success