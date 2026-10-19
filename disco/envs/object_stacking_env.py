"""
Object stacking environment loaded from a saved environment.yaml, with tower checks.
"""

import copy
import os
import numpy as np
import torch
from omegaconf import OmegaConf

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import (
    DefaultMaterialsConfig,
    SceneConfig,
    SimConfig,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from disco.envs.env_utils import ManiSkillEnvUtils
from disco.envs.robot_controller import PANDA_HOME_QPOS, RobotController
from disco.motion.action_primitives import ActionPrimitives
from disco.utils.common import to_sapien_pose, to_torch_pose
from disco.utils.conversion_utils import (
    move_pose_along_local_z,
)


EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13


@register_env("ObjectStacking-v1", max_episode_steps=10000)
class ObjectStacking(BaseEnv, ManiSkillEnvUtils):
    """Tabletop env where objects are stacked into a single tower."""

    def __init__(
        self,
        *args,
        env_path,
        env_index=0,
        **kwargs,
    ):
        """Load the env's scene yaml and build the ManiSkill env.

        Args:
            *args: Forwarded to BaseEnv.
            env_path: Folder holding the env_<i>/ scene folders.
            env_index: Which env_<i> to load.
            **kwargs: Forwarded to BaseEnv (e.g. num_envs, which must be 32).
        """
        self.env_path = env_path
        self.env_index = env_index

        yaml_path = os.path.join(env_path, f"env_{env_index}", "environment.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"environment.yaml not found at {yaml_path}")
        self.scene_data = OmegaConf.load(yaml_path)
        self.env_folder = os.path.join(env_path, f"env_{env_index}")
        self.num_objects = len(self.scene_data.objects)

        self.robot_init_qpos_noise = 0.0
        self.scene_bounds = [-0.3, -0.3, 0.0, 0.3, 0.3, 0.5]

        # Per-search root base index. Set by the planner before tower_height
        # is queried; left None when env is used standalone.
        self._mcts_root_base_idx = None

        super().__init__(*args, robot_uids="panda", **kwargs)

    # ---------------- ManiSkill required configs ---------------- #

    @property
    def _default_sim_config(self):
        """Return physics settings tuned for stable stacking of small mesh objects."""
        # See TableBussing comment: small mesh objects + VHACD multi-hulls
        # need a smaller contact skin and PCM disabled for stable stacking.
        return SimConfig(
            spacing=3,
            sim_freq=240,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.005,
                rest_offset=0.0,
                solver_position_iterations=30,
                solver_velocity_iterations=2,
                bounce_threshold=4.0,
                enable_pcm=False,
            ),
            default_materials_config=DefaultMaterialsConfig(
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        )

    @property
    def _default_sensor_configs(self):
        """Return the single 256x256 observation camera."""
        pose = sapien_utils.look_at(eye=[0.4, 0.4, 0.6], target=[0.0, 0.0, 0.05])
        return [CameraConfig("camera_1", pose=pose, width=256, height=256, fov=np.pi / 2)]

    @property
    def _default_human_render_camera_configs(self):
        """Return the 512x512 render camera."""
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.6], target=[0.0, 0.0, 0.05])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.2, near=0.01, far=100,
        )

    # ---------------- Scene loading ---------------- #

    def _load_agent(self, options: dict):
        """Load the Panda robot.

        Args:
            options: ManiSkill reconfigure options.
        """
        super()._load_agent(options)

    def _load_scene(self, options: dict):
        """Build the table and load each object's URDF from the scene yaml.

        Args:
            options: ManiSkill reconfigure options.
        """
        self.scene_builder = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.scene_builder.build()

        self.actors = []
        self.movable_actors = []
        self.table = self.scene_builder.table
        self.actors.append(self.table)

        self.urdf_loader = self.scene.create_urdf_loader()
        self.urdf_loader.load_multiple_collisions_from_file = True
        self.urdf_loader.disable_self_collisions = True

        self.objects = {}
        self.object_ids = {}
        self.object_names = []
        self.object_scale_factors = {}
        self.object_poses = {}
        # Per-object local mesh AABB dims cached from yaml `bounds`.
        # Used by get_object_current_dims so we don't go through
        # ManiSkill's get_collision_meshes (which has a multi-env
        # device-mismatch bug on `cuda_rigid_body_data`).
        self._local_dims_cache = {}

        object_items = sorted(
            self.scene_data.objects.items(),
            key=lambda kv: int(kv[0].rsplit("_", 1)[1]),
        )

        for i, (slot_name, props) in enumerate(object_items):
            urdf_path = os.path.join(self.env_folder, props.urdf_filename)
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"URDF not found for {slot_name}: {urdf_path}")

            # NOTE: URDFs in `assets/environments/*/<split>/env_<i>/` embed the
            # mesh scale already (see scripts/env_editing/* output). Do NOT
            # set `urdf_loader.scale` here or objects double-scale to
            # sub-millimeter sizes. scale_factor is kept as metadata only.
            scale = float(props.scale_factor)
            pose = np.array(list(props.pose), dtype=np.float32)

            # Use object_id from yaml (the bare 32-char hash) so that
            # object_hash(name) produces the same key as the grasp-cache
            # files — matching the shelf_packing_env convention.
            object_name = f"{props.object_id}_{i}"
            self.object_names.append(object_name)
            self.object_scale_factors[object_name] = scale

            actor_builders = self.urdf_loader.parse(urdf_path)["actor_builders"]
            builder = actor_builders[0]

            self.object_poses[object_name] = pose
            builder.initial_pose = to_sapien_pose(pose)

            self.objects[object_name] = builder.build(name=object_name)
            self.object_ids[object_name] = self.objects[object_name].per_scene_id[0].item()
            self.actors.append(self.objects[object_name])
            self.movable_actors.append(self.objects[object_name])

            # Cache local AABB dims from yaml `bounds`.
            # bounds shape: [[xmin, xmax], [ymin, ymax], [zmin, zmax]] in
            # mesh-local frame (already scaled — yaml is post-scale).
            if "bounds" in props:
                b = np.array(props.bounds, dtype=np.float64)
                self._local_dims_cache[object_name] = (
                    float(b[0, 1] - b[0, 0]),
                    float(b[1, 1] - b[1, 0]),
                    float(b[2, 1] - b[2, 0]),
                )

    # ---------------- Episode init ---------------- #

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset the table, robot, and object poses, and create the robot controller.

        Args:
            env_idx: Indices of the env slots being reset.
            options: ManiSkill reset options.
        """
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            self.agent.reset(np.tile(PANDA_HOME_QPOS, (len(env_idx), 1)))
            for object_name in self.object_names:
                self.set_object_pose(object_name, self.object_poses[object_name])

            self.robot_controller = RobotController(
                self,
                debug=False,
                vis=False,
                base_pose=self.unwrapped.agent.robot.pose[0],
                visualize_target_grasp_pose=True,
                print_env_info=False,
            )

    def setup_primitives(self):
        """Cache initial object and home poses, and create the cuRobo action primitives.

        Call after env.reset(), so the robot controller exists.
        """
        self.initial_object_poses = {}
        for object_name in self.object_names:
            self.initial_object_poses[object_name] = copy.deepcopy(
                self.get_object_pose(object_name)
            )

        self.home_pose = self.get_gripper_pose().clone()
        self.home_pose = move_pose_along_local_z(
            self.home_pose, -EE_LINK_CENTER_TO_GRIPPER_TIP,
        )
        self.home_pose = to_torch_pose(self.home_pose)
        self.home_joints = self.get_joint_state().clone()

        self.action_primitives = ActionPrimitives(env=self)

    # ---------------- Object helpers (carried over from ShelfPackingBase) -- #

    def get_object_current_dims(self, object_name):
        """Return an object's local (dx, dy, dz) bounding-box size in meters.

        Args:
            object_name: Object name.
        """
        # Prefer the cached yaml-derived AABB to avoid
        # ManiSkill's `get_collision_meshes()` path, which calls
        # `actor.pose` -> `cuda_rigid_body_data.torch()[scene_idxs]`
        # with mixed CPU/GPU index tensors when num_envs > 1.
        cached = self._local_dims_cache.get(object_name)
        if cached is not None:
            return cached
        object_mesh = self.objects[object_name].get_collision_meshes()[0]
        dx = object_mesh.vertices[:, 0].max() - object_mesh.vertices[:, 0].min()
        dy = object_mesh.vertices[:, 1].max() - object_mesh.vertices[:, 1].min()
        dz = object_mesh.vertices[:, 2].max() - object_mesh.vertices[:, 2].min()
        return float(dx), float(dy), float(dz)

    def get_object_contact_norms(self, object_name):
        """Return the net contact force magnitude on an object, per env slot.

        Args:
            object_name: Object name, or the actor itself.

        Returns:
            (num_envs,) tensor.
        """
        if isinstance(object_name, str):
            contact_forces = self.objects[object_name].get_net_contact_forces()
        else:
            contact_forces = object_name.get_net_contact_forces()
        return torch.norm(contact_forces, dim=-1)

    # ---------------- Stacking predicates ---------------- #

    @staticmethod
    def _world_half_extents(quat_wxyz: torch.Tensor, dims) -> torch.Tensor:
        """Return the (E, 3) world-axis half extents of a rotated box.

        Args:
            quat_wxyz: (E, 4) box orientations, wxyz.
            dims: Local (dx, dy, dz) box size.
        """
        w, x, y, z = quat_wxyz.unbind(-1)
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ], -2)                                                         # (E, 3, 3)
        half = torch.tensor(dims, dtype=R.dtype, device=R.device) / 2
        return (R.abs() * half).sum(-1)

    # Tower predicate. "pose" (default) is the pose-only rule used for the stacking metric; "contact"
    # additionally requires pairwise contact and uses rotated extents, and under-counts towers that stand.
    stack_predicate = "pose"
    STACK_TOL = 0.05

    def stacked_matrix(self, contact_threshold: float = 0.01) -> torch.Tensor:
        """Return a (num_envs, n, n) bool matrix; [e, top, bot] means top rests on bot in slot e.

        Args:
            contact_threshold: Min pairwise contact force; used only by the "contact" predicate.
        """
        if self.stack_predicate == "pose":
            return self._stacked_matrix_pose()
        return self._stacked_matrix_contact(contact_threshold)

    def _stacked_matrix_pose(self) -> torch.Tensor:
        """Return the stacked matrix from object poses and local dims only.

        Top's xy centre must lie within bot's footprint radius, above bot's top face, within STACK_TOL.
        """
        n = self.num_objects
        poses = self.object_poses_tensor                          # (E, n, 7)
        dims = torch.tensor([self.get_object_current_dims(nm) for nm in self.object_names],
                            dtype=poses.dtype, device=poses.device)  # (n, 3)
        S = torch.zeros((self.num_envs, n, n), dtype=torch.bool, device=poses.device)
        for b in range(n):
            radius = torch.maximum(dims[b, 0], dims[b, 1]) / 2
            for t in range(n):
                if t == b:
                    continue
                xy = torch.linalg.norm(poses[:, t, :2] - poses[:, b, :2], dim=-1) < radius
                dz = poses[:, t, 2] - poses[:, b, 2]
                S[:, t, b] = xy & (dz > dims[b, 2] / 2) & (dz < dims[b, 2] / 2 + dims[t, 2] / 2 + self.STACK_TOL)
        return S

    def _stacked_matrix_contact(self, contact_threshold: float = 0.01) -> torch.Tensor:
        """Return the stacked matrix from pairwise contact and rotated object extents.

        Args:
            contact_threshold: Min pairwise contact force for two objects to count as touching.
        """
        n = self.num_objects
        poses = self.object_poses_tensor                          # (E, n, 7)
        dims = [self.get_object_current_dims(nm) for nm in self.object_names]
        S = torch.zeros((self.num_envs, n, n), dtype=torch.bool, device=poses.device)
        for b in range(n):
            ext = self._world_half_extents(poses[:, b, 3:], dims[b])         # (E, 3)
            radius = torch.maximum(ext[:, 0], ext[:, 1])
            for t in range(n):
                if t == b:
                    continue
                f = self.scene.get_pairwise_contact_forces(
                    self.objects[self.object_names[t]], self.objects[self.object_names[b]],
                )
                contact = torch.linalg.norm(f.reshape(self.num_envs, -1), dim=-1) > contact_threshold
                xy = torch.linalg.norm(poses[:, t, :2] - poses[:, b, :2], dim=-1) < radius
                above = poses[:, t, 2] > poses[:, b, 2] + ext[:, 2]
                S[:, t, b] = contact.to(S.device) & xy & above
        return S

    def is_stacked_on(self, top_idx: int, bot_idx: int, contact_threshold: float = 0.01,
                      env_idx: int = 0) -> bool:
        """Return whether object top_idx rests on object bot_idx.

        Args:
            top_idx: Index of the upper object.
            bot_idx: Index of the lower object.
            contact_threshold: Passed to stacked_matrix.
            env_idx: Env slot to check.
        """
        if top_idx == bot_idx:
            return False
        return bool(self.stacked_matrix(contact_threshold)[env_idx, top_idx, bot_idx])

    def _chain_up(self, S_env, base: int):
        """Return the longest stacked chain of object indices upward from base, base-first.

        Args:
            S_env: (n, n) stacked matrix of one env slot.
            base: Index of the bottom object.
        """
        best = [base]
        for t in range(self.num_objects):
            if bool(S_env[t, base]):
                chain = [base] + self._chain_up(S_env, t)
                if len(chain) > len(best):
                    best = chain
        return best

    def tower_height(self, root_base_idx: int, env_idx: int = 0):
        """Return (height, chain) of the tallest tower built on a given base, chain base-first.

        Args:
            root_base_idx: Index of the base object.
            env_idx: Env slot to check.
        """
        chain = self._chain_up(self.stacked_matrix()[env_idx].cpu(), root_base_idx)
        return len(chain), chain

    def tallest_tower(self, env_idx: int = 0, S: torch.Tensor = None):
        """Return (height, chain) of the tallest tower over all bases, chain base-first.

        Args:
            env_idx: Env slot to check.
            S: Precomputed stacked matrix; computed if None.
        """
        S_env = (self.stacked_matrix() if S is None else S)[env_idx].cpu()
        chain = max((self._chain_up(S_env, b) for b in range(self.num_objects)), key=len)
        return len(chain), chain

    def stack_success(self, moved_idx: int, base_idx: int, pre_base_pose: torch.Tensor,
                      trans_tol: float = 0.01, quat_tol: float = 0.05,
                      env_idx: int = 0):
        """Check that a placed object rests on its base without knocking the base.

        Args:
            moved_idx: Index of the placed object.
            base_idx: Index of the base object.
            pre_base_pose: (7,) base pose before the place.
            trans_tol: Max base translation drift in meters.
            quat_tol: Max base rotation drift, as 1 - |q_before . q_after|.
            env_idx: Env slot to check.

        Returns:
            (ok, failure_type); failure_type is None, "no_contact_or_offset",
            "on_table" or "knocked_base".
        """
        if not self.is_stacked_on(moved_idx, base_idx, env_idx=env_idx):
            return False, "no_contact_or_offset"

        moved_z = self.get_object_pose(self.object_names[moved_idx])[env_idx, 2].item()
        if moved_z < 0.02:
            return False, "on_table"

        cur_base_pose = self.get_object_pose(self.object_names[base_idx])[env_idx]
        pre = pre_base_pose.to(cur_base_pose.device).to(cur_base_pose.dtype)
        trans_drift = torch.norm(cur_base_pose[:3] - pre[:3]).item()
        quat_dot = torch.dot(cur_base_pose[3:], pre[3:]).item()
        quat_drift = 1.0 - abs(quat_dot)
        if trans_drift > trans_tol or quat_drift > quat_tol:
            return False, "knocked_base"
        return True, None

    # ---------------- ManiSkill API stubs ---------------- #

    def evaluate(self):
        """Return per-slot tallest tower height and success (every object in one tower)."""
        S = self.stacked_matrix()
        h = torch.tensor([self.tallest_tower(e, S)[0] for e in range(self.num_envs)],
                         dtype=torch.float32, device=self.device)
        return {"success": h >= self.num_objects, "tower_height": h}

    def compute_dense_reward(self, obs=None, action=None, info=None):
        """Return the (num_envs,) tallest tower height.

        Args:
            obs: Unused.
            action: Unused.
            info: Output of evaluate(); computed if None.
        """
        # ManiSkill calls with (obs, action, info); ActionPrimitives calls
        # zero-arg. Cover both: returns the live per-env tallest-tower height
        # as a (num_envs,) tensor.
        if info is None:
            info = self.evaluate()
        return info["tower_height"]

    def compute_normalized_dense_reward(self, obs=None, action=None, info=None):
        """Return the (num_envs,) tallest tower height divided by the object count.

        Args:
            obs: Unused.
            action: Unused.
            info: Output of evaluate(); computed if None.
        """
        if info is None:
            info = self.evaluate()
        return info["tower_height"] / max(self.num_objects, 1)

    # ---- Shelf-compat stubs (Pick/Place primitives + shared helpers
    # occasionally reach for these on the env; for stacking they have
    # no meaningful answer). ----

    def number_of_objects_in_shelf(self, env_idx: int = 0):
        """Return (0, []), since stacking has no shelf.

        Args:
            env_idx: Unused.
        """
        return 0, []

    def check_object_inside_shelf(self, object_name, env_idx: int = 0):
        """Return False, since stacking has no shelf.

        Args:
            object_name: Unused.
            env_idx: Unused.
        """
        return False
