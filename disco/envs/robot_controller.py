"""
Panda arm and gripper control and trajectory execution in the 32 parallel sims.
"""
import mplib
import numpy as np
import sapien
import torch
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose

from disco.utils.common import build_panda_gripper_grasp_pose_visual

import sapien.physx as physx
OPEN = 1
CLOSED = -1



# Robot home configuration (7 arm joints + 2 finger joints). Applied after ManiSkill's
# TableSceneBuilder resets the Panda (its own default home differs), so stock ManiSkill works.
PANDA_HOME_QPOS = np.array([-1.3159, -0.4246, 0.1067, -2.7110, -0.0562, 2.3219, 0.7518, 0.04, 0.04])


class RobotController:
    """Drive the Panda arm and gripper in the sim: trajectory execution plus mplib planning helpers."""
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        """Wrap the env's Panda robot and set up an mplib planner for it.

        Args:
            env: ManiSkill env (32 slots) whose robot is controlled.
            debug: Wait for a key press in the viewer after mplib plans (needs vis).
            vis: Render the human viewer after each step.
            base_pose: Robot base pose in the world.
            visualize_target_grasp_pose: Show a gripper marker at mplib target poses (needs vis).
            print_env_info: Print reward and info after each step.
            joint_vel_limits: Per-joint velocity limit for mplib.
            joint_acc_limits: Per-joint acceleration limit for mplib.
        """
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.gripper_state_batch = np.full((self.env.num_envs, 1), self.gripper_state)
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_panda_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            # self.set_grasp_pose_visual(self.base_env.agent.tcp.pose)
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    
    def set_grasp_pose_visual(self, pose: sapien.Pose):
        """Move the target-grasp marker, if any, to a pose.

        Args:
            pose: Target gripper pose in the world.
        """
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
            current_action = self.env.get_robot_qpos()
            _ = self.env.control_robot(current_action)
            # self.env.capture_video_frame()

    def render_wait(self):
        """Block in the viewer until the user presses c (only when vis and debug are on)."""
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        """Return an mplib planner for the robot's hand TCP, placed at the base pose."""
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        """Step the env through an mplib plan with a single shared action (unbatched).

        Args:
            result: mplib plan dict with "position" (and "velocity") arrays.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            (obs, reward, terminated, truncated, info) of the last step.
        """
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def follow_path_from_position(self, position_path, refine_steps: int = 0):
        """Step the env through a joint-position path with a single shared action (unbatched).

        Args:
            position_path: (T, 7) joint positions.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            (obs, reward, terminated, truncated, info) of the last step.
        """
        n_step = position_path.shape[0]
        for i in range(n_step + refine_steps):
            qpos = position_path[min(i, n_step - 1)]
            action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def follow_path_from_position_parallel(self, position_path, refine_steps: int = 0):
        """Step every sim slot through its own joint-position path.

        Args:
            position_path: (num_envs, T, 7) joint positions.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            (obs, reward, terminated, truncated, info) of the last step.
        """
        n_step = position_path.shape[1]
        
        for i in range(n_step + refine_steps):
            step_idx = min(i, n_step - 1)
            qpos = position_path[:, step_idx, :]  # Shape: (num_envs, 7)
            
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state_batch])  # Shape: (num_envs, 8)
            else:
                zeros = np.zeros_like(qpos)  # Shape: (num_envs, 7)
                action = np.hstack([qpos, zeros, self.gripper_state_batch])  # Shape: (num_envs, 15)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def execute_trajectory(self, trajectories, gripper_steps: int = None, refine_steps: int = 0):
        """Execute joint + gripper trajectories in the first B sim slots; the other slots hold still.

        Whenever a row's gripper command changes, the arm holds while the gripper actuates.

        Args:
            trajectories: (B, T, 8) numpy array of 7 joint positions plus gripper (1 open, -1 closed), B <= num_envs.
            gripper_steps: Sim steps per gripper change; None uses env.gripper_steps or 6.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            (obs, reward, terminated, truncated, info) of the last step.
        """
        assert self.control_mode == "pd_joint_pos", "Only pd_joint_pos control mode supported"
        if gripper_steps is None:  # per-env override (e.g. a slower release for stacking), default 6
            gripper_steps = getattr(self.base_env, "gripper_steps", 6)

        batch_size, n_step, _ = trajectories.shape
        num_envs = self.env.num_envs
        
        # Assert that batch_size is not greater than num_envs
        assert batch_size <= num_envs, f"trajectories batch_size ({batch_size}) cannot be greater than num_envs ({num_envs})"
        
        joint_positions = trajectories[:, :, :7]  # (batch_size, T, 7)
        gripper_states = trajectories[:, :, 7]    # (batch_size, T)
        
        # Track current gripper state for all environments
        current_gripper = self.gripper_state_batch.flatten().copy()  # (num_envs,)
        
        # Get current robot state for all environments (for environments not in batch)
        current_qpos = self.robot.get_qpos()[:, :-2].cpu().numpy()  # (num_envs, 7)
        
        for i in range(n_step + refine_steps):
            step_idx = min(i, n_step - 1)
            target_qpos = joint_positions[:, step_idx, :]  # (batch_size, 7)
            new_gripper = gripper_states[:, step_idx]  # (batch_size,)
            
            # Prepare action for all environments
            # For environments in batch: use trajectory
            # For environments not in batch: use current state (no movement)
            action_qpos = current_qpos.copy()  # (num_envs, 7)
            action_qpos[:batch_size] = target_qpos  # Update first batch_size environments
            
            action_gripper = current_gripper.copy()  # (num_envs,)
            action_gripper[:batch_size] = new_gripper  # Update first batch_size environments
            
            # Execute trajectory step
            action = np.hstack([action_qpos, action_gripper[:, np.newaxis]])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            # Render via the standard env.render() path (gated on render_mode)
            # to match the convention used elsewhere in the codebase.
            if self.base_env.render_mode is not None:
                self.base_env.render()

            # Check if any environment in batch has a gripper state change
            batch_gripper_changed = (new_gripper != current_gripper[:batch_size]).any()
            if batch_gripper_changed:
                current_gripper[:batch_size] = new_gripper.copy()
                # Get robot's CURRENT position and hold it while actuating gripper
                hold_qpos = self.robot.get_qpos()[:, :-2].cpu().numpy()  # (num_envs, 7)
                for _ in range(gripper_steps):
                    action = np.hstack([hold_qpos, current_gripper[:, np.newaxis]])
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self.elapsed_steps += 1
                    if self.print_env_info:
                        print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
                    if self.base_env.render_mode is not None:
                        self.base_env.render()
        
        # Update internal gripper state
        self.gripper_state_batch = current_gripper[:, np.newaxis].copy()
        
        return obs, reward, terminated, truncated, info

    def execute_trajectory_with_recording(self, trajectories, gripper_steps: int = 6, refine_steps: int = 0):
        """Execute trajectories like execute_trajectory, recording the sim state after every step.

        Args:
            trajectories: (B, T, 8) numpy array of 7 joint positions plus gripper command, B <= num_envs.
            gripper_steps: Sim steps per gripper change.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            Last-step (obs, reward, terminated, truncated, info), then lists of recorded states and sent actions.
        """
        assert self.control_mode == "pd_joint_pos", "Only pd_joint_pos control mode supported"

        batch_size, n_step, _ = trajectories.shape
        num_envs = self.env.num_envs

        assert batch_size <= num_envs, f"trajectories batch_size ({batch_size}) cannot be greater than num_envs ({num_envs})"

        joint_positions = trajectories[:, :, :7]
        gripper_states = trajectories[:, :, 7]

        current_gripper = self.gripper_state_batch.flatten().copy()
        current_qpos = self.robot.get_qpos()[:, :-2].cpu().numpy()

        recorded_states = []
        recorded_actions = []

        from tqdm import tqdm

        # Count total steps for tqdm
        total_steps = n_step + refine_steps
        # Estimate extra gripper steps (can't know exact count ahead of time)
        pbar = tqdm(total=total_steps, desc="Recording trajectory", leave=False)

        for i in range(n_step + refine_steps):
            step_idx = min(i, n_step - 1)
            target_qpos = joint_positions[:, step_idx, :]
            new_gripper = gripper_states[:, step_idx]

            action_qpos = current_qpos.copy()
            action_qpos[:batch_size] = target_qpos

            action_gripper = current_gripper.copy()
            action_gripper[:batch_size] = new_gripper

            action = np.hstack([action_qpos, action_gripper[:, np.newaxis]])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1

            recorded_actions.append(action.copy())
            recorded_states.append(self.env.get_state().clone())

            if self.vis:
                self.base_env.render_human()

            pbar.update(1)

            batch_gripper_changed = (new_gripper != current_gripper[:batch_size]).any()
            if batch_gripper_changed:
                current_gripper[:batch_size] = new_gripper.copy()
                hold_qpos = self.robot.get_qpos()[:, :-2].cpu().numpy()
                pbar.total += gripper_steps
                pbar.refresh()
                for _ in range(gripper_steps):
                    action = np.hstack([hold_qpos, current_gripper[:, np.newaxis]])
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self.elapsed_steps += 1

                    recorded_actions.append(action.copy())
                    recorded_states.append(self.env.get_state().clone())

                    if self.vis:
                        self.base_env.render_human()
                    pbar.update(1)

        pbar.close()

        self.gripper_state_batch = current_gripper[:, np.newaxis].copy()

        return obs, reward, terminated, truncated, info, recorded_states, recorded_actions

    def follow_path_from_position_until_gripper_collision(self, position_path, refine_steps: int = 0):
        """Step through a joint-position path (unbatched) until the gripper collides.

        Args:
            position_path: (T, 7) joint positions.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            The part of the path before the collision, or the whole path.
        """
        n_step = position_path.shape[0]
        for i in range(n_step + refine_steps):
            qpos = position_path[min(i, n_step - 1)]
            action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            gripper_collision = self.env.check_gripper_collision()
            if gripper_collision:
                return_path_until = min(i, n_step - 1) - 1
                return position_path[:return_path_until]
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return position_path

    def follow_path_from_position_until_gripper_collision_parallel(self, position_path, refine_steps: int = 0):
        """Step every slot through its own joint path, freezing each slot once its gripper collides.

        Args:
            position_path: (num_envs, T, 7) joint positions.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            (num_envs,) path index where each slot stopped (one before its collision).
        """
        num_envs = position_path.shape[0]
        n_step = position_path.shape[1]
        
        # Track which environments have collided (torch tensor)
        has_collided = torch.zeros(num_envs, dtype=torch.bool, device=self.env.device)
        # Track current step index for each environment (torch tensor)
        current_step = torch.zeros(num_envs, dtype=torch.long, device=self.env.device)
        # Track where each environment stopped (torch tensor)
        stop_indices = torch.full((num_envs,), n_step + refine_steps - 1, dtype=torch.long, device=self.env.device)
        
        # Initialize with first position
        qpos = position_path[:, 0, :]  # Shape: (num_envs, 7)
        
        for i in range(n_step + refine_steps):
            # For environments that haven't collided, advance to next step
            # For environments that have collided, keep using the same position
            step_idx = torch.clamp(current_step, max=n_step - 1)
            step_idx_np = step_idx.cpu().numpy()
            qpos = position_path[np.arange(num_envs), step_idx_np, :]  # Shape: (num_envs, 7)
            
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state_batch])  # Shape: (num_envs, 8)
            else:
                zeros = np.zeros_like(qpos)  # Shape: (num_envs, 7)
                action = np.hstack([qpos, zeros, self.gripper_state_batch])  # Shape: (num_envs, 15)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            
            # Check collisions for all environments
            gripper_collisions = self.env.check_gripper_collision()  # Shape: (num_envs,) torch bool
            
            # Record stop index for environments that just collided (before updating has_collided)
            newly_collided = gripper_collisions & ~has_collided
            collision_step = min(i, n_step - 1)
            stop_indices[newly_collided] = max(collision_step - 1, 0)  # NOTE: -1 because we want to stop before the collision.
            # Keep current_step at the collision point for newly collided environments
            current_step[newly_collided] = collision_step
            
            # Update collision status (once True, stays True)
            has_collided = has_collided | gripper_collisions
            
            # Advance step index only for environments that haven't collided
            current_step[~has_collided] += 1
            
            # Stop early if all environments have collided
            if has_collided.all():
                break
            
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        
        return stop_indices.cpu().numpy()

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Plan to a TCP pose with mplib RRTConnect from slot 0's joints and follow the path.

        Args:
            pose: Target TCP pose in the world.
            dry_run: Return the plan without executing it.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            The mplib plan if dry_run, -1 if planning failed, else the last step's outputs.
        """
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.set_grasp_pose_visual(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Plan a straight-line screw motion to a TCP pose with mplib and follow it.

        Tries twice; a screw plan fails on collision instead of going around obstacles.

        Args:
            pose: Target TCP pose in the world.
            dry_run: Return the plan without executing it.
            refine_steps: Extra steps holding the final waypoint.

        Returns:
            The mplib plan if dry_run, -1 if planning failed, else the last step's outputs.
        """
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.set_grasp_pose_visual(pose)
        pose = sapien.Pose(p=pose.p , q=pose.q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def control_gripper(self, gripper_states, num_steps=6):
        """Set the gripper command in every slot and step with the arm held in place.

        Args:
            gripper_states: Scalar, (num_envs,) or (num_envs, 1) gripper commands (1 open, -1 closed).
            num_steps: Sim steps to run.

        Returns:
            (obs, reward, terminated, truncated, info) of the last step.
        """
        qpos = self.robot.get_qpos()[:, :-2].cpu().numpy()  # Shape: (num_envs, 7)
        num_envs = qpos.shape[0]
        
        # Convert gripper_states to numpy array and expand dimensions if necessary
        gripper_states = np.asarray(gripper_states)
        if gripper_states.ndim == 0:
            gripper_states = np.full((num_envs, 1), gripper_states.item())
        if gripper_states.ndim == 1:
            # 1D array: ensure it matches num_envs
            if gripper_states.shape[0] != num_envs:
                raise ValueError(f"gripper_states shape {gripper_states.shape} does not match num_envs {num_envs}")
            gripper_state_batch = gripper_states[:, np.newaxis]  # Shape: (num_envs, 1)
        elif gripper_states.ndim == 2:
            # 2D array: ensure it matches num_envs
            if gripper_states.shape[0] != num_envs:
                raise ValueError(f"gripper_states shape {gripper_states.shape} does not match num_envs {num_envs}")
            if gripper_states.shape[1] == 1:
                gripper_state_batch = gripper_states  # Shape: (num_envs, 1)
            else:
                raise ValueError(f"gripper_states must have shape (num_envs,) or (num_envs, 1), got {gripper_states.shape}")
        else:
            raise ValueError(f"gripper_states must be scalar, 1D, or 2D array, got {gripper_states.ndim}D")
        
        for i in range(num_steps):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, gripper_state_batch])  # Shape: (num_envs, 8)
            else:
                zeros = np.zeros_like(qpos)  # Shape: (num_envs, 7)
                action = np.hstack([qpos, zeros, gripper_state_batch])  # Shape: (num_envs, 15)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        self.gripper_state_batch = gripper_state_batch.copy()
        return obs, reward, terminated, truncated, info

    def open_gripper(self):
        """Open the gripper over 6 sim steps, holding slot 0's arm joints (unbatched).

        Returns:
            (obs, reward, terminated, truncated, info) of the last step.
        """
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state = CLOSED):
        """Close the gripper over t sim steps, holding slot 0's arm joints (unbatched).

        Args:
            t: Number of sim steps.
            gripper_state: Gripper command to apply (default closed).

        Returns:
            (obs, reward, terminated, truncated, info) of the last step.
        """
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        """Add a box, sampled as surface points, to the mplib collision point cloud.

        Args:
            extents: (3,) box side lengths (m).
            pose: Box pose in the world.
        """
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        """Add points to the mplib collision point cloud.

        Args:
            pts: (N, 3) xyz points in the world.
        """
        self.use_point_cloud = True
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        """Forget the stored collision points and stop using the point cloud in mplib plans."""
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        """Do nothing; kept for interface compatibility."""
        pass
