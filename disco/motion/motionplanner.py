"""
cuRobo motion planner: IK, batched motion planning and the collision world.
"""

import os

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig, Mesh
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import Obstacle
from curobo.wrap.reacher.motion_gen import PoseCostMetric, MotionGenResult


# --------------------------------------------------------------------------- #
# cuRobo batch copy_idx shape-robustness patch (does NOT edit the submodule).
#
# In plan_batch's multi-attempt loop, `_plan_batch_attempts` accumulates the
# best result across attempts via `best_result.copy_idx(idx, result)`, which
# does `current_tensor[idx] = source_tensor[idx]` on the per-goal error tensors
# (position_error / rotation_error / cspace_error). Across attempts cuRobo
# returns these inconsistently shaped — sometimes (B,) , sometimes (B, 1), and
# at batch 1 even 0-dim — so the assignment raises:
#   "too many indices for tensor of dimension 0" / "shape mismatch ... [2] vs [2,1]".
# These error tensors are not used by our trajectory extraction (we read only
# optimized_plan.position + success), so we make the index-copy shape-robust:
# reshape the source slice to the destination slice when element counts match,
# else skip that field. Patches the staticmethod at runtime; the vendored
# cuRobo file is untouched.
# --------------------------------------------------------------------------- #
# Class access on a staticmethod already yields the plain function.
_curobo_orig_check_copy_idx = MotionGenResult._check_none_and_copy_idx


def _safe_check_none_and_copy_idx(current_tensor, source_tensor, idx):
    """Copy source_tensor[idx] into current_tensor[idx], skipping cuRobo's mismatched shapes.

    Args:
        current_tensor: Destination tensor holding the best result so far.
        source_tensor: Tensor from the latest planning attempt.
        idx: Indices to copy.

    Returns:
        The updated current_tensor.
    """
    if (source_tensor is not None and current_tensor is not None
            and isinstance(current_tensor, torch.Tensor)
            and isinstance(source_tensor, torch.Tensor)):
        try:
            dst = current_tensor[idx]
            src = source_tensor[idx]
            if dst.shape != src.shape:
                if dst.numel() == src.numel():
                    src = src.reshape(dst.shape)
                else:
                    return current_tensor  # incompatible; skip (error fields unused)
            current_tensor[idx] = src
            return current_tensor
        except IndexError:
            # 0-dim / un-indexable error tensor (batch-size-1 squeeze). These
            # fields are unused downstream, so skip rather than crash.
            return current_tensor
    return _curobo_orig_check_copy_idx(current_tensor, source_tensor, idx)


MotionGenResult._check_none_and_copy_idx = staticmethod(_safe_check_none_and_copy_idx)


torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import trimesh
from trimesh.creation import icosphere, axis
from disco.utils.conversion_utils import (
    pose_to_transformation, 
    move_pose_along_local_z, 
    transformation_to_pose, 
    invert_transformation
)
from disco.utils.common import to_torch_pose
import copy
from tqdm import tqdm

EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

# Constants for motion planning
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Obstacle meshes are exported here (absolute paths are passed to cuRobo).
CUROBO_MESH_DIR = os.path.join(_REPO, "assets", "curobo_meshes")
# Upstream cuRobo Franka config with this project's collision-sphere settings.
CUROBO_ROBOT_CONFIG = os.path.join(_REPO, "config", "curobo", "franka.yml")
POST_GRASP_LIFT = 0.15
GRASP_DEPTH = 0.25

class MotionPlanner:
    """cuRobo motion planner and IK solver over a world of object-mesh obstacles."""

    def __init__(
        self, 
        mesh_file_dict: dict = None, 
        initial_object_poses: dict = None,
        ):
        """Build the cuRobo world from the object meshes and warm up the planner.

        Args:
            mesh_file_dict: Obstacle name -> mesh file path.
            initial_object_poses: Obstacle name -> (7,) initial world pose (xyz, wxyz quat).
        """

        self.mesh_file_dict = mesh_file_dict
        self.initial_object_poses = initial_object_poses

        self.reset_planner()

        self.links = [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
            "attached_object",
        ]
        
    def reset_planner(self):
        """Create the cuRobo world (object meshes plus a back wall) and a warmed-up MotionGen."""

        setup_curobo_logger("error")
        tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(CUROBO_ROBOT_CONFIG)

        # Create world configuration with individual obstacles
        self.world_config = WorldConfig()

        for mesh_name, mesh_path in self.mesh_file_dict.items():
            obstacle = Mesh(
                file_path=mesh_path,
                name=mesh_name,
                pose=[0.615, 0, 0, 1, 0, 0, 0]
            )
            self.world_config.add_obstacle(obstacle)

        self.back_wall = Cuboid(
            name = "back_wall",
            pose = [-0.4, 0., 0.5, 1, 0, 0, 0],
            dims = [0.2, 1.4, 1.0]
        )
        self.world_config.add_obstacle(self.back_wall)

        # self.front_wall = Cuboid(
        #     name = "front_wall",
        #     pose = [0.85, 0., 0.5, 1, 0, 0, 0],
        #     dims = [0.2, 1.4, 1.0]
        # )
        # self.world_config.add_obstacle(self.front_wall)

        # # This is from robot perspective
        # self.right_wall = Cuboid(
        #     name = "right_wall",
        #     pose = [0.5, -(0.52 + 0.2/2), 0.5, 1, 0, 0, 0],
        #     dims = [1., 0.2, 1.0]
        # )
        # self.world_config.add_obstacle(self.right_wall)
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            self.world_config,
            tensor_args,
            interpolation_dt=0.01,
            # trajopt_dt=0.15,
            # velocity_scale=0.1,
            use_cuda_graph=False,
            # finetune_dt_scale=2.5,
            interpolation_steps=1000,   # was 10000
        )        

        self.motion_gen = MotionGen(motion_gen_config)
        # Warmup with goalset planning since we use plan_grasp which requires goalset
        # Set n_goalset to the maximum number of grasps you expect to use (Basically the maximum number of poses that can be passed to plan_goalset)
        
        # self.motion_gen.warmup(enable_graph=True, n_goalset=100)
        self.motion_gen.warmup(n_goalset=200)

    def sapien_pose_to_curobo_pose(self, sapien_pose):

        """Convert a world-frame gripper-tip pose to the robot-frame EE-link pose cuRobo uses.

        Args:
            sapien_pose: (7,) or (N, 7) world-frame pose (xyz, wxyz quat).

        Returns:
            Pose tensor on cuda:0, shifted +0.615 m in x and backed off 0.13 m along local Z.
        """
        sapien_pose = to_torch_pose(sapien_pose)
        curobo_pose = copy.deepcopy(sapien_pose)
        curobo_pose[..., 0] += 0.615       # TODO: Manually adding 0.615 to grasp pose, to transform it to robot base frame, this needs to be automatic
        curobo_pose = move_pose_along_local_z(curobo_pose, -EE_LINK_CENTER_TO_GRIPPER_TIP)
        curobo_pose = to_torch_pose(curobo_pose, device="cuda:0")

        return curobo_pose

    def curobo_pose_to_sapien_pose(self, curobo_pose: torch.Tensor):
        """Convert a robot-frame EE-link pose from cuRobo back to a world-frame gripper-tip pose.

        Args:
            curobo_pose: (7,) or (N, 7) robot-frame pose tensor (xyz, wxyz quat).

        Returns:
            Numpy pose array in the sim world frame.
        """
        sapien_pose = copy.deepcopy(curobo_pose.squeeze().cpu().numpy())
        sapien_pose[..., 0] -= 0.615       # TODO: Manually adding 0.615 to grasp pose, to transform it to robot base frame, this needs to be automatic
        sapien_pose = move_pose_along_local_z(sapien_pose, EE_LINK_CENTER_TO_GRIPPER_TIP)
        return sapien_pose
       
    def set_collision_world_components(
        self,  
        enable: bool,
        objects: list[str] = [], 
        collision_links: list[str] = []
        ):
        """Enable or disable collision checks for named obstacles and robot links.

        Args:
            enable: True to enable checks, False to disable them.
            objects: World obstacle names to toggle.
            collision_links: Robot link names to toggle; all other links are re-enabled.
        """
        if len(collision_links) > 0:
            self.motion_gen.toggle_link_collision(collision_links, enable)
            # Enable all other links that are not in the collision_links list
            other_links = [link for link in self.links if link not in collision_links]
            if len(other_links) > 0:
                self.motion_gen.toggle_link_collision(other_links, True)

        for object_name in objects:
            self.motion_gen.world_coll_checker.enable_obstacle(enable=enable, name=object_name)
        
    def clear_gpu_memory(self):
        """Free cached CUDA memory and reset the IK solver after planning."""
        import torch
        import gc
        
        # Clear PyTorch's CUDA cache
        torch.cuda.empty_cache()
        
        # Optionally reset cuRobo's internal buffers (less aggressive)
        # self.motion_gen.reset(reset_seed=False)
        
        # Force Python garbage collection
        gc.collect()
        
        # Clear cache again after garbage collection
        torch.cuda.empty_cache()

        # Reset cuRobo's internal state (add this)
        self.motion_gen.ik_solver.solver.reset()

    
    def plan_to_joint_state(self, current_joint_state: torch.Tensor, goal_joint_state: torch.Tensor, holding_object: bool = False):
        """Plan a single joint-space trajectory to a goal configuration.

        Args:
            current_joint_state: (7,) start joint positions.
            goal_joint_state: (7,) goal joint positions.
            holding_object: If False, ignore collisions of the attached-object link.

        Returns:
            (interpolated joint trajectory or None, success flag).
        """
        
        start_joint = JointState.from_position(current_joint_state.view(1, -1))
        goal_joint = JointState.from_position(goal_joint_state.view(1, -1))
        
        plan_config = MotionGenPlanConfig(
            max_attempts=100,
        )

        if not holding_object:
            disable_collision_links = ["attached_object"]
            self.motion_gen.toggle_link_collision(disable_collision_links, False)

        result = self.motion_gen.plan_single_js(start_joint, goal_joint, plan_config)
        self.clear_gpu_memory()

        success = result.success.item()
        if success:
            return result.get_interpolated_plan().position, success
        else:
            return None, success
    
    # ==================================================== #

    # ================ PARALLEL PLANNING FUNCTIONS MULTI OBJECT ================ #

    def inverse_kinematics(
        self, 
        input_grasp_poses: torch.Tensor, 
        objects_to_ignore: list[str] = [], 
        disable_collision_links: list[str] = ["attached_object"]
        ):
        """Solve collision-aware IK for a batch of end-effector poses.

        Args:
            input_grasp_poses: (N, 7) robot-frame poses (xyz, wxyz quat), N <= 1000.
            objects_to_ignore: Obstacle names excluded from collision checks.
            disable_collision_links: Robot links excluded from collision checks.

        Returns:
            (N, 7) joint solutions (zeros where IK failed) and (N, 1) boolean success mask.
        """

        # Disable collision checking for relevant world components
        self.set_collision_world_components(
            enable=False, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )
        
        # Predefine the output joint states tensor:
        ik_solutions = torch.zeros((input_grasp_poses.shape[0], 7), device=input_grasp_poses.device, dtype=torch.float32)
        
        # Convert to Pose
        ik_poses = Pose(position=input_grasp_poses[..., :3], quaternion=input_grasp_poses[..., 3:])
        
        # Solve IK for the batch
        ik_result = self.motion_gen.ik_solver.solve_batch(ik_poses)

        # Get successful indices and solutions, then store them in the output tensor:
        success = ik_result.success.to(input_grasp_poses.device)    # (batch_size,) boolean tensor
        valid_indices = torch.where(success)[0]
        if len(valid_indices) > 0:
            ik_solutions[valid_indices] = ik_result.solution[valid_indices, 0]
        else:
            print(f"No valid IK solutions found for the poses")

        # Re-enable collision checking for relevant world components
        self.set_collision_world_components(
            enable=True, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )

        self.clear_gpu_memory()
            
        return ik_solutions, success
    
    def plan_to_goal_poses(
        self, 
        current_joints: torch.Tensor, 
        goal_poses: torch.Tensor, 
        objects_to_ignore: list[str] = [], 
        disable_collision_links: list[str] = ["attached_object"],
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(max_attempts=100)
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan batched trajectories from start joints to goal end-effector poses.

        Args:
            current_joints: (N, 7) start joint positions, N <= 50.
            goal_poses: (N, 7) robot-frame goal poses (xyz, wxyz quat).
            objects_to_ignore: Obstacle names excluded from collision checks.
            disable_collision_links: Robot links excluded from collision checks.
            plan_config: cuRobo plan settings (attempts, pose constraints).

        Returns:
            (N, T, 7) trajectories (failed rows hold the start joints) and (N,) boolean success mask.
        """
        
        traj_len = self.motion_gen.trajopt_solver.traj_tsteps
        assert current_joints.shape[0] == goal_poses.shape[0], "Current joint states and goal poses must have the same batch size"
        assert current_joints.shape[1] == 7 and goal_poses.shape[1] == 7, "Current joint states and goal poses must have 7 dimensions"
        assert current_joints.ndim == 2 and goal_poses.ndim == 2, "Current joint states and goal poses must be 2D tensors"

        # cuRobo's plan_batch crashes on batch size 1: it squeezes per-goal error
        # tensors (position_error, ...) to 0-dim while keeping `success` 1-D, so
        # its multi-attempt copy_idx accumulation does `position_error[idx]` on a
        # scalar ("too many indices for tensor of dimension 0"). Pad a size-1
        # batch to 2 by duplicating the single query, then drop the duplicate
        # row before returning so callers see the original batch size.
        orig_batch = current_joints.shape[0]
        if orig_batch == 1:
            current_joints = current_joints.repeat(2, 1)
            goal_poses = goal_poses.repeat(2, 1)

        # Disable collision checking for relevant world components
        self.set_collision_world_components(
            enable=False, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )
        
        # Convert to JointState and Pose
        start_states = JointState.from_position(current_joints)
        curobo_goal_poses = Pose(position=goal_poses[..., :3], quaternion=goal_poses[..., 3:])

        # Initialize the solution, by default as just staying at the current joint state:
        trajectories = current_joints.clone()
        trajectories = trajectories.unsqueeze(1).repeat(1, traj_len, 1)  # (N, traj_len, 7)
        
        # NOTE: For cuRobo motion planning, 
        # use interpolated_plan if you need fine-grained waypoints, or optimized_plan if coarse is sufficient
        # Plan for the batch if environments:
        plan_result = self.motion_gen.plan_batch(
            start_state=start_states, 
            goal_pose=curobo_goal_poses, 
            plan_config=plan_config
        )

        # Get successful indices and solutions, then store them in the output tensor:
        success = plan_result.success.to(current_joints.device)    # (batch_size,) boolean tensor
        successful_indices = torch.where(success)[0]

        if len(successful_indices) == 0:
            return trajectories[:orig_batch], success[:orig_batch]

        # cuRobo squeezes the batch dim when batch size == 1: optimized_plan.position
        # comes back (T, 7) instead of (1, T, 7). Indexing that by successful_indices
        # would slice the time axis. Restore the leading batch dim first.
        optimized_position = plan_result.optimized_plan.position
        if optimized_position.ndim == 2:
            optimized_position = optimized_position.unsqueeze(0)  # (1, T, 7)

        trajectories[successful_indices] = optimized_position[successful_indices].clone()

        # Re-enable collision checking for relevant world components
        self.set_collision_world_components(
            enable=True, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )

        self.clear_gpu_memory()

        return trajectories[:orig_batch], success[:orig_batch]

    # ==================================================== #

    def check_pose_collision(self, pose: torch.Tensor):
        """Solve single-pose IK to check that a pose is reachable without collision.

        Args:
            pose: (7,) robot-frame pose (xyz, wxyz quat).

        Returns:
            (joint solution, success flag).
        """
        
        pose = Pose(pose[:3], quaternion=pose[3:])
        result = self.motion_gen.ik_solver.solve_single(pose)
        return result.js_solution.position[0], result.success.item()
    
    def attach_objects_to_robot(self, current_joint_state: torch.Tensor, object_names: list):
        """Attach world obstacles to the gripper as collision spheres.

        Args:
            current_joint_state: (7,) current joint positions.
            object_names: Obstacle names to attach.

        Returns:
            cuRobo success flag.
        """
        joint_state = JointState.from_position(current_joint_state.view(1, -1))
        success = self.motion_gen.attach_objects_to_robot(
            joint_state, 
            object_names,
            sphere_fit_type = SphereFitType.VOXEL_SURFACE,
            surface_sphere_radius=0.015)

        return success
    
    def detach_objects_from_robot(self):
        """Detach all attached objects from the robot and return cuRobo's success flag."""
        success = self.motion_gen.detach_object_from_robot()
        return success
    
    def visualize_world_and_robot(self, q: torch.Tensor = None, pose: torch.Tensor = None):

        """Show the cuRobo world, robot collision spheres, and an optional pose frame in a viewer.

        Args:
            q: (7,) joint configuration; zeros if None.
            pose: Optional (7,) robot-frame pose (xyz, wxyz quat) drawn as an axis triad.
        """

        # Optional: visualize an arbitrary pose as a frame in the scene
        T = None
        if pose is not None:
            if isinstance(pose, torch.Tensor):
                pose_np = pose.detach().cpu().numpy()
            else:
                pose_np = np.asarray(pose, dtype=np.float32)
            # pose_np expected as (x, y, z, qw, qx, qy, qz)
            T = pose_to_transformation(pose_np, format='wxyz')
        
        world = self.world_config
        scene = WorldConfig.get_scene_graph(world)

        # Add a small axis triad to visualize the provided pose
        if T is not None:
            frame = axis(origin_size=0.02, axis_length=0.12, transform=T)
            scene.add_geometry(frame)

        # robot spheres from FK
        if q is None:
            q = torch.zeros(7, dtype=torch.float32, device="cuda:0")
        if q.dim() == 1:
            q = q.view(1, -1)
        kin = self.motion_gen.compute_kinematics(JointState.from_position(q))
        spheres = kin.robot_spheres.squeeze(0).cpu().numpy()  # [n,4] x,y,z,r

        for x, y, z, r in spheres:
            s = icosphere(subdivisions=2, radius=float(r))
            s.apply_translation([float(x), float(y), float(z)])
            s.visual.face_colors = [200, 50, 50, 120]
            scene.add_geometry(s)

        scene.show()
    
    def fk(self, q: torch.Tensor, link_name: str = None):
        """Compute the end-effector pose for joint configurations with cuRobo kinematics.

        Args:
            q: (7,) or (B, 7) joint positions.
            link_name: Link to report instead of the end effector, if given.

        Returns:
            (7,) or (B, 7) robot-frame pose (xyz, wxyz quat).
        """
        if q.dim() == 1:
            q = q.view(1, -1)
        ee_pose = torch.zeros_like(q)
        fk_result = self.motion_gen.ik_solver.kinematics.get_state(q, link_name=link_name)
        ee_pose[:, :3] = fk_result.ee_position
        ee_pose[:, 3:] = fk_result.ee_quaternion
        if q.shape[0] == 1:
            ee_pose = ee_pose.squeeze(0)
        return ee_pose
