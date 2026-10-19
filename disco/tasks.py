"""
Task adapters: what differs between shelf packing and object stacking.
"""
import gymnasium as gym
import numpy as np

from disco.utils.pcd_sampling import (
    downsample_pcd_with_mask, downsample_pcd_xyz, filter_pcd_target_only,
    sample_scene_with_obj_ids, shelved_object_indices,
)

NUM_POINTS = 4096
# Failures a robot's planner reports before moving; any other failure type is an execution outcome.
PLANNING_FAILURES = {"ik_infeasible", "pre_grasp_motion_plan", "grasp_motion_plan",
                     "lift_motion_plan", "place_motion_plan", "no_base"}


def _scene(env, exclude):
    """Sample slot 0's scene cloud without the table, ground, robot and excluded objects.

    Args:
        env: Simulator.
        exclude: Object indices to drop.

    Returns:
        (N, 3) world-frame xyz and (N,) object index per point (-1 = non-object actor).
    """
    xyz, ids = sample_scene_with_obj_ids(env, env_idx=0, extra_ignore=("table",))
    keep = ~np.isin(ids, np.asarray(sorted(exclude), dtype=np.int32))
    return xyz[keep], ids[keep]


class Task:
    """Base adapter shared by the shelf and stacking tasks."""
    name = ""
    env_id = ""
    policy_channels = 3
    env_kwargs = {}
    gripper_steps = 6

    def make_env(self, env_root, env_index, num_envs=32):
        """Build, reset and settle the task's simulator, with motion primitives set up.

        Args:
            env_root: Directory of env scenes (env_<i>/).
            env_index: Env to load.
            num_envs: Parallel sim slots; must stay 32.

        Returns:
            The unwrapped ManiSkill env.
        """
        import disco.envs.object_stacking_env  # noqa: F401  (registers ObjectStacking-v1)
        import disco.envs.shelf_packing_env  # noqa: F401  (registers ShelfPacking-v1)
        env = gym.make(self.env_id, num_envs=num_envs, obs_mode="state",
                       control_mode="pd_joint_pos", render_mode=None,
                       env_path=env_root, env_index=env_index, **self.env_kwargs)
        env.reset(seed=[0] * num_envs)
        u = env.unwrapped
        u.wait_for_synchronized_stability()
        u.setup_primitives()
        u.gripper_steps = self.gripper_steps  # sim steps per gripper open / close (execute_trajectory default)
        return u

    def value_cloud(self, env, exclude=()):
        """Return the full-scene cloud for the value network Q.

        Args:
            env: Simulator (slot 0 is read).
            exclude: Object indices to drop.

        Returns:
            (4096, 3) float32 world-frame xyz.
        """
        xyz, _ = _scene(env, exclude)
        return downsample_pcd_xyz(xyz, num_points=NUM_POINTS).astype(np.float32)

    def settle(self, env):
        """Let the scene settle after a state is set with set_state; no-op by default.

        Args:
            env: Simulator.
        """


class ShelfTask(Task):
    """Shelf packing: move every object from the table into the shelf."""
    name = "shelf"
    env_id = "ShelfPacking-v1"
    policy_channels = 3
    env_kwargs = {"parallel_in_single_scene": False}

    def movable_objects(self, env):
        """Return the indices of objects not yet inside the shelf (slot 0).

        Args:
            env: Simulator.
        """
        return [i for i, n in enumerate(env.object_names)
                if not env.check_object_inside_shelf(n, env_idx=0)]

    def policy_cloud(self, env, obj, exclude=()):
        """Return the policy cloud: target object, shelf and shelved objects (other table objects removed).

        Args:
            env: Simulator (slot 0 is read).
            obj: Index of the target object.
            exclude: Object indices to drop.

        Returns:
            (4096, 3) float32 world-frame xyz.
        """
        xyz, ids = _scene(env, exclude)
        tgt = filter_pcd_target_only(xyz, ids, obj, shelved_object_indices(env, env_idx=0))
        return downsample_pcd_xyz(tgt, num_points=NUM_POINTS).astype(np.float32)

    def plan_place(self, env, place_params):
        """Motion-plan shelf placements of the held object (no execution).

        Args:
            env: Simulator holding the post-grasp states.
            place_params: (n, 8) rows of [place pose7, object index].
        """
        return env.action_primitives.plan_place(place_params)

    def judge_place(self, env, plan):
        """Return the outcome of an executed shelf place plan.

        Args:
            env: Simulator whose first rows hold the plan's end states.
            plan: Plan from `plan_place`.
        """
        return env.action_primitives.judge_place(plan)

    def progress(self, env, env_idx=0):
        """Return the fraction of objects inside the shelf.

        Args:
            env: Simulator.
            env_idx: Sim slot to read.
        """
        n_in, _ = env.number_of_objects_in_shelf(env_idx=env_idx)
        return float(n_in) / env.num_objects

    def reward(self, env):
        """Return slot 0's packed-volume fraction (the MCTS edge reward is its change).

        Args:
            env: Simulator.
        """
        return float(env.compute_dense_reward()[0])


class StackingTask(Task):
    """Object stacking: build one tower of all objects."""
    name = "stacking"
    env_id = "ObjectStacking-v1"
    policy_channels = 4
    base_margin = 0.02

    def movable_objects(self, env):
        """Return the indices of objects outside the tallest tower (all objects if no tower of 2+).

        Args:
            env: Simulator (slot 0 is read).
        """
        h, chain = env.tallest_tower(0)
        if h < 2:
            return list(range(env.num_objects))
        return [i for i in range(env.num_objects) if i not in chain]

    def policy_cloud(self, env, obj, exclude=()):
        """Return the policy cloud: all objects, xyz plus a target-object mask channel.

        Args:
            env: Simulator (slot 0 is read).
            obj: Index of the target object.
            exclude: Object indices to drop.

        Returns:
            (4096, 4) float32 world-frame cloud.
        """
        xyz, ids = _scene(env, exclude)
        return downsample_pcd_with_mask(xyz, (ids == obj).astype(np.float32),
                                        num_points=NUM_POINTS).astype(np.float32)

    def infer_bases(self, env, place_params):
        """Return, per place row, the object the held object will rest on (-1 for none).

        Base = highest other object below the object's place pose whose xy footprint (+margin) contains it.

        Args:
            env: Simulator with each row in its post-grasp state.
            place_params: (n, 8) rows of [gripper place pose7, object index].

        Returns:
            List of n base object indices.
        """
        from disco.motion.action_primitives import EE_LINK_CENTER_TO_GRIPPER_TIP
        from disco.utils.pose_utils import compose_pose, inv_pose, quat_rotate
        poses = env.object_poses_tensor.cpu().numpy().astype(np.float64)   # (E, n, 7)
        tcp = env.get_gripper_pose().cpu().numpy().astype(np.float64)     # (E, 7) fingertip (TCP) frame
        moved = int(round(float(place_params[0, 7])))
        bases = []
        for r in range(place_params.shape[0]):
            # Grasp / place poses use the EE-link frame, EE_LINK_CENTER_TO_GRIPPER_TIP behind the TCP
            # along its local z (same convention as setup_primitives' home pose).
            ee = tcp[r].copy()
            ee[:3] = tcp[r, :3] + quat_rotate(tcp[r, 3:], np.array([0.0, 0.0, -EE_LINK_CENTER_TO_GRIPPER_TIP]))
            g_to_o = compose_pose(inv_pose(ee), poses[r, moved])
            obj_place = compose_pose(place_params[r, :7].cpu().numpy().astype(np.float64), g_to_o)
            best = -1
            for b in range(env.num_objects):
                if b == moved or poses[r, b, 2] >= obj_place[2]:
                    continue
                dx, dy, _ = env.get_object_current_dims(env.object_names[b])
                d = float(np.linalg.norm(obj_place[:2] - poses[r, b, :2]))
                if d < max(dx, dy) / 2 + self.base_margin and (best < 0 or poses[r, b, 2] > poses[r, best, 2]):
                    best = b
            bases.append(best)
        return bases

    free_yaw = False  # PlaceStacking: allow any rotation about world z at the place pose
    descend = 0.03    # PlaceStacking: lower this far (m) from the place pose before releasing (as recorded)
    lift_fallback = True   # PlaceStacking: retry a failed post-release lift (as the executed plans were recorded)
    gripper_steps = 25     # slow release: the executed stacking plans were recorded with 25

    def plan_place(self, env, place_params):
        """Motion-plan stacking placements of the held object onto the inferred bases (no execution).

        Args:
            env: Simulator holding the post-grasp states.
            place_params: (n, 8) rows of [place pose7, object index].
        """
        return env.action_primitives.plan_place_stacking(
            place_params, base_idx=self.infer_bases(env, place_params), free_yaw=self.free_yaw,
            descend=self.descend, lift_fallback=self.lift_fallback)

    def judge_place(self, env, plan):
        """Return the outcome of an executed stacking place plan.

        Args:
            env: Simulator whose first rows hold the plan's settled end states.
            plan: Plan from `plan_place`.
        """
        return env.action_primitives.judge_place_stacking(plan)

    def progress(self, env, env_idx=0):
        """Return the tallest tower height as a fraction of all objects.

        Args:
            env: Simulator.
            env_idx: Sim slot to read.
        """
        return float(env.tallest_tower(env_idx)[0]) / env.num_objects

    def reward(self, env):
        """Return slot 0's tallest-tower fraction (the MCTS edge reward is its change).

        Args:
            env: Simulator.
        """
        return self.progress(env)

    def settle(self, env):
        """Wait up to 50 sim steps for the objects to come to rest.

        Args:
            env: Simulator.
        """
        env.wait_for_stability(max_steps=50)


def get_task(name):
    """Return a new task adapter.

    Args:
        name: "shelf" or "stacking".
    """
    return {"shelf": ShelfTask, "stacking": StackingTask}[name]()
