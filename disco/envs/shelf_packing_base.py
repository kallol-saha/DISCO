"""
Base shelf packing environment: scene, objects, shelf checks and rewards.
"""
import json
import os
import copy
import time
import sapien
from sympy.core import I
import torch
import shutil
import numpy as np
from typing import Tuple
from omegaconf import OmegaConf

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import articulations, actors
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from sapien.physx import PhysxRigidBodyComponent

from disco.envs.robot_controller import PANDA_HOME_QPOS, RobotController
from disco.motion.action_primitives import ActionPrimitives
from disco.policy.shelf_sampler import ActionSampler
from disco.envs.env_utils import ManiSkillEnvUtils

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from tqdm import tqdm
from disco.utils.common import (
    load_grasps,
    to_numpy_pose,
    to_sapien_pose,
    to_torch_pose,
    flip_quat_about_z,
    flip_pose_quat_about_z,
    load_acronym_object_and_grasps,
    point_in_cuboid,
    generate_placement_quaternions
)

from disco.utils.visualization.plotting import (
    plot_pcd,
    plot_pcd_with_highlighted_segment
)
from disco.utils.visualization.point_cloud_structures import make_gripper_visualization
from disco.utils.conversion_utils import (
    pose_to_transformation,
    invert_transformation,
    transform_pcd,
    transformation_to_pose,
    move_pose_along_local_z,
    move_pose_along_local_x,
    move_pose_along_local_y,
    furthest_point_sample,
    downsample_point_cloud
)
from disco.flowmatch.modeling.policy.value_network import ValueNetwork
from disco.flowmatch.utils.common_utils import count_parameters

from disco.envs.generation_utils import (
    compute_ray_box_intersection
)

POST_GRASP_LIFT = 0.15
GRASP_DEPTH = 0.25
GRASP_RETRACTION_DISTANCE = 0.15
EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

CUROBO_SUCCESSES = 0

OPEN = 1
CLOSED = -1


def load_checkpoint_for_eval(checkpoint_path, model):
    """Load checkpoint weights into a model, reporting missing or unexpected keys.

    Args:
        checkpoint_path: Path to a .pth file holding a "weight" state dict.
        model: Module to load the weights into (non-strict).

    Returns:
        The same model with the loaded weights.
    """
    print("=> trying checkpoint '{}'".format(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    try:
        model_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
    except Exception as e:
        if "numpy.core.multiarray.scalar" in str(e):
            print(f"Warning: Falling back to weights_only=False due to numpy scalar in checkpoint: {e}")
            model_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False
            )
            # Convert numpy scalars to Python types if present
            if "best_loss" in model_dict and hasattr(model_dict["best_loss"], "item"):
                model_dict["best_loss"] = float(model_dict["best_loss"].item())
        else:
            raise
    # Load weights flexibly

    msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
    if msn:
        print(f"Missing keys (not found in checkpoint): {len(msn)}")
        print(msn)
    if unxpct:
        print(f"Unexpected keys (ignored): {len(unxpct)}")
        print(unxpct)
    if not msn and not unxpct:
        print("All keys matched successfully!")

    print("=> loaded successfully '{}' (step {})".format(
        checkpoint_path, model_dict.get("iter", 0)
    ))
    del model_dict
    torch.cuda.empty_cache()

    return model


class ShelfPackingBase(BaseEnv, ManiSkillEnvUtils):
    """Shelf-packing ManiSkill env: table, shelf, Panda robot, and packing checks."""

    # ---------------- INITIALIZATION ---------------- #

    def __init__(
        self,
        *args,
        num_objects=1,
        model_path=None,
        policy_type="placement",
        policy_device="cuda:0",
        value_model_path=None,
        **kwargs):
        """Build the env, then optionally load a sampling policy and a value network.

        Args:
            *args: Forwarded to ManiSkill BaseEnv.
            num_objects: Number of objects (subclasses reset it from the env file).
            model_path: Policy checkpoint to load into the action sampler, or None.
            policy_type: Policy kind for ActionSampler.load_policy (e.g. "placement").
            policy_device: Device to put the policy on.
            value_model_path: Value-network checkpoint, or None.
            **kwargs: Forwarded to ManiSkill BaseEnv (e.g. num_envs).
        """

        self.robot_init_qpos_noise = 0.
        self.action_primitives = None

        self.num_objects = num_objects

        self.scene_bounds = [-0.4, -0.5, 0.01, 0.3, 0.5, 0.8]
        self.camera_bounds = [-0.4, -0.75, 0.01, 0.45, 0.75, 0.8]
        self.pcd_bounds = [-0.4, -0.8, 0.01, 0.45, 0.8, 0.8]

        super().__init__(*args, robot_uids="panda", **kwargs)

        # The sampler owns the policy. Build it first so we have somewhere
        # to load the policy into; load policies as opt-in.
        self.action_sampler = ActionSampler(self)

        if model_path is not None:
            self.model_path = model_path
            self.action_sampler.load_policy(
                policy_type=policy_type,
                model_path=model_path,
                device=policy_device,
            )

        if value_model_path is not None:
            self.value_model_path = value_model_path
            self.load_value_function()

    @property
    def placement_model(self):
        """Return the action sampler's placement policy (back-compat alias)."""
        # Back-compat shim. Old code reads `env.placement_model` directly;
        # the model now lives on the action sampler. New code should read
        # `env.action_sampler.placement_model`.
        return self.action_sampler.placement_model

    def init_value_function(self):
        """Build an untrained value network on the GPU in eval mode."""
        print("Initializing fresh value function (no checkpoint)")
        self.value_model = ValueNetwork(
            embedding_dim=120,
            num_attn_heads=8,
            num_shared_attn_layers=4,
        )

        count_parameters(self.value_model)

        for name, param in self.value_model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.value_model.to(device)
        self.value_model.eval()

    def load_value_function(self):
        """Build the value network and load weights from `self.value_model_path`."""
        print(f"Loading value function from {self.value_model_path}")
        self.init_value_function()
        self.value_model = load_checkpoint_for_eval(self.value_model_path, self.value_model)

    @property
    def _default_sim_config(self):
        """Return the sim config (3 m spacing between parallel scenes)."""
        return SimConfig(spacing=3)

    @property
    def _default_sensor_configs(self):
        """Return two 256x256 cameras, one facing the shelf and one the table objects.

        Also caches the shelf front center and both camera poses/transforms.
        """
        self.shelf_front_center = np.array([
                self.shelf_pose[0] - (self.shelf_depth / 2) * np.cos(self.shelf_rotation),
                self.shelf_pose[1] - (self.shelf_depth / 2) * np.sin(self.shelf_rotation),
                self.shelf_distance_from_floor + self.wall_thickness + self.shelf_height/2
            ])

        # cam1: right side of the scene (+Y), looking at the shelf.
        shelf_cam_point = np.array([
            self.shelf_front_center[0],
            self.camera_bounds[4],
            self.shelf_front_center[2],
        ])

        # cam2: front of the scene (−Y), looking at the objects on the table.
        object_cam_look_at = np.array([-0.065, 0., 0.05])
        object_cam_point = np.array([
            object_cam_look_at[0],
            self.camera_bounds[1],
            object_cam_look_at[2] + 0.4,
        ])

        self.camera_1_pose = sapien_utils.look_at(
            eye=shelf_cam_point,
            target=self.shelf_front_center
        )
        self.camera_2_pose = sapien_utils.look_at(
            eye=object_cam_point,
            target=object_cam_look_at
        )

        self.camera_1_transform = pose_to_transformation(self.camera_1_pose.raw_pose[0].cpu().numpy(), format='wxyz')
        self.camera_2_transform = pose_to_transformation(self.camera_2_pose.raw_pose[0].cpu().numpy(), format='wxyz')

        # return [
        #     CameraConfig(
        #         "camera_1",
        #         pose=self.camera_1_pose,
        #         width=256,
        #         height=256,
        #         fov= np.pi/2,
        #     ),
        #     CameraConfig(
        #         "camera_2",
        #         pose=self.camera_2_pose,
        #         width=256,
        #         height=256,
        #         fov= np.pi/2,
        #     ),
        # ]

        return [
            CameraConfig(
                "camera_1",
                pose=self.camera_1_pose,
                width=256,
                height=256,
                fov= np.pi/2,
            ),
            CameraConfig(
                "camera_2",
                pose=self.camera_2_pose,
                width=256,
                height=256,
                fov= np.pi/2,
            ),
        ]


    @property
    def _default_human_render_camera_configs(self):
        """Return the 512x512 render camera aimed at the shelf front."""
        width = 512
        height = 512
        pose = sapien_utils.look_at(
            eye = [0.3, -0.75, self.shelf_front_center[2] + 0.2],
            target = self.shelf_front_center
        )
        self.video_frames = np.zeros((0, width, height, 3))
        return CameraConfig(
            "render_camera", pose=pose, width=width, height=height, fov=1.2, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        """Load the Panda at its fixed base pose and cache base transform and gripper links.

        Args:
            options: ManiSkill load options, passed to the parent loader.
        """
        self.robot_base_pose = sapien.Pose(p=[0.615, 0, 0], q=[1, 0, 0, 0])
        self.world_to_robot_base_transform = pose_to_transformation(
            np.concatenate([self.robot_base_pose.p, self.robot_base_pose.q]),
            format='wxyz'
            )
        self.world_to_robot_base_transform = torch.tensor(self.world_to_robot_base_transform, dtype=torch.float32, device=self.device)
        super()._load_agent(options, self.robot_base_pose)

        self.robot_links = self.agent.robot.get_links()
        self.gripper_link_names = self.robot_link_names[-6:]

    def _initialize_scene(self):
        """Load the objects and shelf into the scene; subclasses must implement."""
        raise NotImplementedError("Subclasses must implement _initialize_scene()")

    def _load_scene(self, options: dict):
        """Build the table, load objects and shelf, and cache the shelf interior box.

        Args:
            options: ManiSkill load options (unused).
        """

        # Load in the table:
        self.scene_builder = TableSceneBuilder(env = self,
                                             robot_init_qpos_noise = self.robot_init_qpos_noise)
        self.scene_builder.build()

        self.actors = []
        self.movable_actors = []

        self.table = self.scene_builder.table
        self.table_id = self.table.per_scene_id[0].item()
        self.actors.append(self.table)

        # Load in the urdf loader
        self.urdf_loader = self.scene.create_urdf_loader()
        self.urdf_loader.load_multiple_collisions_from_file = True

        # Initialize object properties dictionary
        self.objects = {}
        self.object_ids = {}
        self.grasps = {}
        self.bounds = {}
        self.grasp_costs = {}
        self.object_quaternions = {}
        self.object_scale_factors = {}
        self.object_poses = {}
        self.object_angles = {}
        self.object_volumes = {}

        print("")

        self._initialize_scene()

        x_min = -self.shelf_depth/2
        x_max = self.shelf_depth/2
        y_min = -self.shelf_width/2
        y_max = self.shelf_width/2
        z_min = self.shelf_distance_from_floor + self.wall_thickness/2
        z_max = z_min + self.shelf_height

        # For calculating bounds:
        self.shelf_corners = np.array(
            [[x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_min, y_max, z_min],
            [x_max, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_min, y_max, z_max],
            [x_max, y_max, z_max]])

        shelf_transform = pose_to_transformation(self.shelf_pose, format='wxyz')
        self.transform_to_axis_aligned_shelf = invert_transformation(shelf_transform)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset robot, shelf, and object poses, and create the robot controller.

        Args:
            env_idx: Indices of the env slots being reset.
            options: ManiSkill reset options (unused).
        """
        with torch.device(self.device):

            self.scene_builder.initialize(env_idx)
            self.agent.reset(np.tile(PANDA_HOME_QPOS, (len(env_idx), 1)))

            # Set the shelf pose:
            self.set_object_pose(self.shelf, self.shelf_pose)

            # Set the object poses:
            if hasattr(self, 'initial_object_poses'):
                for object_name in self.object_names:
                    self.set_object_pose(object_name, self.initial_object_poses[object_name])
            else:
                for object_name in self.object_names:
                    self.set_object_pose(object_name, self.object_poses[object_name])

            # Initialize the robot controller. NOTE: vis=False is REQUIRED here.
            # When vis=True, RobotController.__init__ adds a `grasp_pose_visual`
            # actor to the scene (disco/utils/common.py:build_panda_gripper_grasp_pose_visual).
            # _initialize_episode runs AFTER Scene._setup → physx.gpu_init(), and
            # GPU PhysX preallocates buffers per registered actor, so adding an
            # actor at this point invalidates the GPU state. The next
            # _gpu_apply_all() then crashes with "GPU PhysX is not initialized".
            # If you want per-step viewer refreshes during execute_trajectory,
            # flip `env.unwrapped.robot_controller.vis = True` AFTER env init —
            # by then the flag is only used for render_human() calls, not for
            # actor building.
            self.robot_controller = RobotController(
                self,
                debug=False,
                vis=False,
                base_pose=self.unwrapped.agent.robot.pose[0],
                visualize_target_grasp_pose=True,
                print_env_info=False,
            )

    def _before_control_step(self):
        """Do nothing before each control step."""
        pass

    def _after_control_step(self):
        """Render after each control step when a render mode is set."""
        if self.render_mode is not None:
            self.render()

    def setup_primitives(self):
        """Record initial object poses and the home pose, then build the motion primitives."""
        self.initial_object_poses = {}
        for object_name in self.object_names:
            object_pose = self.get_object_pose(object_name)
            self.initial_object_poses[object_name] = copy.deepcopy(object_pose)

        self.home_pose = self.get_gripper_pose().clone()
        self.home_pose = move_pose_along_local_z(self.home_pose, -EE_LINK_CENTER_TO_GRIPPER_TIP)
        self.home_pose = to_torch_pose(self.home_pose)
        self.home_joints = self.get_joint_state().clone()

        self.action_primitives = ActionPrimitives(env=self)

    def get_object_current_dims(self, object_name):
        """Return the object's world-axis x, y, z extents in slot 0.

        Args:
            object_name: Name of the object.
        """
        object_mesh = self.objects[object_name].get_collision_meshes()[0]
        object_dim_x = object_mesh.vertices[:, 0].max() - object_mesh.vertices[:, 0].min()
        object_dim_y = object_mesh.vertices[:, 1].max() - object_mesh.vertices[:, 1].min()
        object_dim_z = object_mesh.vertices[:, 2].max() - object_mesh.vertices[:, 2].min()
        return object_dim_x, object_dim_y, object_dim_z

    # ------------------------------------------------------------------------- #

    # ---------------- TASK SPECIFIC SAMPLING AND VERIFICATION ---------------- #

    # ------------------------------------------------------------------------- #

    def check_object_within_gripper(self, object_name, env_idx = 0):

        """Return True if both gripper fingers touch the object in one env slot.

        Args:
            object_name: Name of the object.
            env_idx: Env slot to check.
        """
        object_actor = self.objects[object_name]
        robot_links = self.agent.robot.get_links()

        left_finger = robot_links[-4]
        right_finger = robot_links[-3]

        left_contact_forces = self.scene.get_pairwise_contact_forces(object_actor, left_finger)[env_idx]
        right_contact_forces = self.scene.get_pairwise_contact_forces(object_actor, right_finger)[env_idx]

        contact_threshold = 0.01
        left_contact = torch.norm(left_contact_forces, dim=-1) > contact_threshold
        right_contact = torch.norm(right_contact_forces, dim=-1) > contact_threshold

        is_grasped = left_contact & right_contact

        return is_grasped.item()

    def check_gripper_collision(self):
        """Return a (num_envs,) bool tensor, True where any gripper link has contact force."""
        gripper_contact_forces = self.agent.robot.get_net_contact_forces(self.gripper_link_names)
        contact_threshold = 0.01
        contact_norms = torch.norm(gripper_contact_forces, dim=-1)
        return (contact_norms > contact_threshold).any(dim=-1)

    def get_object_contact_norms(self, object_name):
        """Return the per-slot magnitude of the net contact force on an object.

        Args:
            object_name: Object name or actor.

        Returns:
            (num_envs,) tensor of contact-force norms.
        """
        if  isinstance(object_name, str):
            contact_forces = self.objects[object_name].get_net_contact_forces()
        else:
            contact_forces = object_name.get_net_contact_forces()

        contact_norms = torch.norm(contact_forces, dim=-1)
        return contact_norms

    def check_object_inside_shelf(self, object_name, env_idx = 0):
        """Return True if the object's center lies inside the shelf interior box.

        Args:
            object_name: Object name or actor.
            env_idx: Env slot to check.
        """

        obj_position = self.get_object_pose(object_name)[env_idx, :3].unsqueeze(0)
        axis_aligned_obj_position = transform_pcd(obj_position.cpu().numpy(), self.transform_to_axis_aligned_shelf)
        return point_in_cuboid(self.shelf_corners, axis_aligned_obj_position[0])

    def number_of_objects_in_shelf(self, env_idx = 0):
        """Count the objects inside the shelf in one env slot.

        Args:
            env_idx: Env slot to check.

        Returns:
            (count, list of indices into self.object_names).
        """

        N = 0
        object_indices_in_shelf = []
        for i in range(len(self.object_names)):
            if self.check_object_inside_shelf(self.object_names[i], env_idx = env_idx):
                N += 1
                object_indices_in_shelf.append(i)

        return N, object_indices_in_shelf

    def number_of_objects_on_table(self, env_idx = 0):
        """Count the objects inside a fixed table region in one env slot.

        Args:
            env_idx: Env slot to check.

        Returns:
            (count, list of indices into self.object_names).
        """
        x_min = 0.25 - 0.615
        x_max = 0.7 - 0.615
        y_min = -0.5
        y_max = 0.07
        z_min = -0.1
        z_max = 0.4

        N = 0
        object_indices_on_table = []

        for i in range(len(self.object_names)):
            object_name = self.object_names[i]
            obj_position = self.get_object_pose(object_name)[env_idx, :3].cpu().numpy()

            if (x_min <= obj_position[0] <= x_max and
                y_min <= obj_position[1] <= y_max and
                z_min <= obj_position[2] <= z_max):
                N += 1
                object_indices_on_table.append(i)

        return N, object_indices_on_table

    def compute_values(self, start_object_state: torch.Tensor):
        """Score each slot by objects in the shelf plus a bonus for small disturbance.

        Args:
            start_object_state: (num_envs, num_objects, 7) object poses to compare against.

        Returns:
            (num_envs,) tensor: shelved count + 1 / (1 + mean abs pose change).
        """

        current_object_state = self.object_poses_tensor
        all_values = []
        for i in range(self.num_envs):
            N, _ = self.number_of_objects_in_shelf(env_idx = i)
            all_values.append(N)

        all_values = torch.tensor(all_values, device=self.device, dtype=torch.float32)

        disturbances = torch.mean(torch.abs(current_object_state - start_object_state), dim=(1, 2))
        all_values = all_values + 1 / (1 + disturbances)

        return all_values

    def set_state_batch(self, states: torch.Tensor):
        """Write states into the first B env slots and leave the rest unchanged.

        Args:
            states: (B, state_dim) sim states with B <= num_envs.
        """
        batch_size = states.shape[0]

        if batch_size > self.num_envs:
            raise ValueError(
                f"Batch size ({batch_size}) exceeds num_envs ({self.num_envs}). "
                f"Cannot set more states than available environments."
            )

        current_state = self.get_state().clone()
        current_state[:batch_size] = states
        self.set_state(current_state)

    def calculate_mean_and_std_value(self):
        """Return the mean and std, across env slots, of the number of shelved objects."""

        all_values = []

        for i in range(self.num_envs):
            value, _ = self.number_of_objects_in_shelf(env_idx = i)
            all_values.append(value)

        mean_value = np.mean(all_values)
        variance_value = np.var(all_values)
        std_value = np.sqrt(variance_value)

        return mean_value, std_value

    # ------------------------------------------------------------------------- #

    # ---------------- VISUALIZATION ---------------- #

    def visualize_policy_predictions(self, out):
        """Plot up to 5 predicted grasp/place gripper pairs over the scene point cloud.

        Args:
            out: (B, 2, >=7) tensor; [:, 0] grasp and [:, 1] place poses (xyz + wxyz quat).
        """
        pair_colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.65, 0.0),
            (0.56, 0.0, 1.0),
        ]

        grasp_pcds = []
        grasp_rgbs = []
        place_pcds = []
        place_rgbs = []

        for i in range(min(5, out.shape[0])):
            pred_grasp_pose = out[i, 0, :7].cpu().detach().numpy()
            pred_place_pose = out[i, 1, :7].cpu().detach().numpy()

            pred_grasp_transform = pose_to_transformation(pred_grasp_pose, format="wxyz")
            pred_place_transform = pose_to_transformation(pred_place_pose, format="wxyz")

            color = pair_colors[i]

            pred_grasp_gripper_pcd, pred_grasp_gripper_colors = make_gripper_visualization(
                rotation=pred_grasp_transform[:3, :3],
                translation=pred_grasp_transform[:3, 3],
                length=0.05,
                density=50,
                color=color,
            )

            pred_place_gripper_pcd, pred_place_gripper_colors = make_gripper_visualization(
                rotation=pred_place_transform[:3, :3],
                translation=pred_place_transform[:3, 3],
                length=0.05,
                density=50,
                color=color,
            )

            grasp_pcds.append(pred_grasp_gripper_pcd)
            grasp_rgbs.append(pred_grasp_gripper_colors)
            place_pcds.append(pred_place_gripper_pcd)
            place_rgbs.append(pred_place_gripper_colors)

        scene_pcd = self.get_sampled_point_cloud()
        scene_rgb = np.ones_like(scene_pcd) * 0.7

        full_pcd = np.vstack([scene_pcd] + grasp_pcds + place_pcds)
        full_rgb = np.vstack([scene_rgb] + grasp_rgbs + place_rgbs)

        plot_pcd(full_pcd, full_rgb)


    def visualize_gripper_poses(self, gripper_poses):
        """Plot gripper poses over the scene point cloud.

        Args:
            gripper_poses: (N, 7) poses (xyz + wxyz quat).
        """
        self.visualize_poses(gripper_poses)

    def sample_object_grasped_poses(self, object_name, n_poses=16):
        """Sample random poses above the table with random yaw, as if held by the gripper.

        Args:
            object_name: Name of the object (unused).
            n_poses: Number of poses to sample.

        Returns:
            (n_poses, 7) tensor of poses (xyz + wxyz quat), z in [0.2, 0.6] m.
        """
        x_min = 0.25
        x_max = 0.7
        y_min = -0.5
        y_max = 0.07
        z_min = 0.2
        z_max = 0.6

        poses = []
        for _ in range(n_poses):
            cx = np.random.uniform(x_min, x_max)
            cy = np.random.uniform(y_min, y_max)
            z = np.random.uniform(z_min, z_max)

            angle = np.random.uniform(0, 2 * np.pi)
            rotation = R.from_euler('z', [angle])
            quaternion = rotation.as_quat().squeeze()

            pose = np.array([
                cx - 0.615,
                cy,
                z,
                quaternion[3],
                quaternion[0],
                quaternion[1],
                quaternion[2]
            ])
            poses.append(pose)

        poses = np.array(poses)
        poses_tensor = torch.tensor(poses, dtype=torch.float32, device=self.device)

        return poses_tensor

    # ------------------------------------------------------------------------- #

    # ---------------- TASK SPECIFIC PLANNING AND EXECUTION ---------------- #

    def test_primitives(self):
        """Smoke-test Pick, Place, and Push across the env slots.

        Legacy: some sampling helpers it calls now live on ActionSampler.

        Returns:
            (grasp_poses, grasp_costs, grasp_object_ids), or None if a step fails.
        """

        grasp_poses, grasp_costs, grasp_object_ids = self.action_sampler.sample_grasp_poses()
        num_grasps = min(self.num_envs, len(grasp_poses))

        trajectories, success = self.action_primitives.Pick(
            grasp_poses=grasp_poses[:num_grasps],
            object_indices=grasp_object_ids[:num_grasps].tolist(),
            lift_distances=torch.tensor([POST_GRASP_LIFT for _ in range(num_grasps)], device="cuda:0", dtype=torch.float32)
        )

        success_indices = np.where(success)[0]
        if len(success_indices) > 0:
            selected_idx = int(success_indices[0])
        else:
            return

        current_state = self.get_state().clone()
        selected_state = current_state[selected_idx].unsqueeze(0)
        current_state[:] = selected_state.expand(self.num_envs, -1)
        self.set_state(current_state)
        if self.render_mode is not None:
            self.render()

        self.action_primitives.update_collision_world()

        object_index = grasp_object_ids[selected_idx].item()

        shelf_poses, shelf_joints = self.sample_gripper_poses_in_shelf(
            n=100,
            object_name=self.object_names[object_index],
            depth_tolerance=0.05,
            width_tolerance=0.05,
            height_tolerance=0.05
        )

        trajectories, success = self.action_primitives.Place(
            place_poses=shelf_poses[:self.num_envs],
            object_index=object_index
        )

        self.action_primitives.OpenGripper()
        self.wait_for_stability()

        object_index = None
        for env_idx in range(self.num_envs):
            num_objects_in_shelf, object_indices_in_shelf = self.number_of_objects_in_shelf(env_idx=env_idx)
            if num_objects_in_shelf > 0:
                object_index = object_indices_in_shelf[0]
                selected_idx = env_idx
                break

        current_state = self.get_state().clone()
        selected_state = current_state[selected_idx].unsqueeze(0)
        current_state[:] = selected_state.expand(self.num_envs, -1)
        self.set_state(current_state)
        if self.render_mode is not None:
            self.render()

        self.action_primitives.update_collision_world()

        if self.render_mode is not None:
            self.render()

        push_start_poses, \
        push_directions, \
        push_distances = self.sample_push_actions(
            object_name=self.object_names[object_index],
            n=100,
            min_push_distance=0.12,
            max_push_distance=0.22,
            max_offset_angle=0.,
            env_idx=0
        )

        self.action_primitives.CloseGripper()

        trajectories, success = self.action_primitives.Push(
            push_start=push_start_poses[:self.num_envs],
            push_direction=push_directions[:self.num_envs],
            push_distance=push_distances[:self.num_envs],
        )

        success_indices = np.where(success)[0]
        if len(success_indices) > 0:
            selected_idx = int(success_indices[0])
        else:
            return

        self.visualize_push_poses(
            push_start_pose = push_start_poses[selected_idx].cpu().numpy(),
            push_direction = push_directions[selected_idx].item(),
            push_distance = push_distances[selected_idx].item()
        )

        return grasp_poses, grasp_costs, grasp_object_ids

    def visualize_push_poses(self, push_start_pose, push_direction, push_distance):
        """Plot the start and end gripper poses of one push.

        Args:
            push_start_pose: (7,) start pose (xyz + wxyz quat).
            push_direction: Push axis in the gripper frame: 0 = x, 1 = y, else z.
            push_distance: Push distance in meters.
        """
        push_start_pose = push_start_pose.reshape(1, -1)
        if push_direction == 0:
            push_end_pose = move_pose_along_local_x(push_start_pose, push_distance, format='wxyz')
        elif push_direction == 1:
            push_end_pose = move_pose_along_local_y(push_start_pose, push_distance, format='wxyz')
        else:
            push_end_pose = move_pose_along_local_z(push_start_pose, push_distance, format='wxyz')

        poses_to_visualize = np.concatenate([push_start_pose, push_end_pose], axis=0)
        self.visualize_grasp_poses(poses_to_visualize)

    def filter_object_poses_in_shelf(self, object_poses):
        """Return indices of object poses whose position is inside the shelf.

        Args:
            object_poses: (N, 7) object poses, numpy array or tensor.
        """
        if isinstance(object_poses, torch.Tensor):
            object_poses_np = object_poses.cpu().numpy()
        else:
            object_poses_np = object_poses

        axis_aligned_obj_points = transform_pcd(object_poses_np[:, :3], self.transform_to_axis_aligned_shelf)

        in_shelf_indices = []
        for i in range(len(axis_aligned_obj_points)):
            if point_in_cuboid(self.shelf_corners, axis_aligned_obj_points[i]):
                in_shelf_indices.append(i)

        return np.array(in_shelf_indices)

    def collect_placement_data(self, batch_size=8, num_points=4096):
        """Collect placement examples: reachable gripper poses that leave the first object in the shelf.

        Legacy: the shelf-pose samplers it calls now live on ActionSampler.

        Args:
            batch_size: Max number of examples (half inside, half outside poses).
            num_points: Points per cloud, split evenly between shelf and object.

        Returns:
            (start grasp poses (B, 7), shelf+object clouds (B, num_points, 3),
            target gripper poses (B, 7)), or False if no valid poses are found.
        """

        object_name = self.object_names[0]

        grasp_poses = self.grasps[object_name]
        grasp_transformation = pose_to_transformation(grasp_poses, format='wxyz')
        grasp_idx = np.random.randint(0, len(grasp_poses))
        grasp_transformation = grasp_transformation[grasp_idx]
        grasp_transform_inverse = invert_transformation(grasp_transformation)

        gripper_pose = self.action_primitives.fk(self.get_joint_state()[0].to("cuda:0")).cpu().detach().numpy()
        gripper_pose[..., 0] = gripper_pose[..., 0] - 0.615
        gripper_transform = pose_to_transformation(gripper_pose, format='wxyz')
        gripper_transform =  gripper_transform @ grasp_transform_inverse
        attached_object_pose = transformation_to_pose(gripper_transform, format='wxyz')
        attached_object_pose = to_torch_pose(attached_object_pose, device="cuda:0")

        self.set_object_pose(object_name, attached_object_pose)
        if self.render_mode is not None:
            self.render()

        self.action_primitives.update_collision_world()
        self.action_primitives.attach_object_to_robot(object_name)

        poses_inside = self.sample_gripper_poses_in_shelf(n=2000)
        poses_outside = self.sample_gripper_poses_outside_shelf(n=2000)

        transforms_inside = pose_to_transformation(poses_inside, format='wxyz')
        transforms_outside = pose_to_transformation(poses_outside, format='wxyz')

        object_transforms_inside = transforms_inside @ grasp_transform_inverse
        object_transforms_outside = transforms_outside @ grasp_transform_inverse

        object_poses_inside = transformation_to_pose(object_transforms_inside, format='wxyz')
        object_poses_outside = transformation_to_pose(object_transforms_outside, format='wxyz')

        object_poses_inside = to_torch_pose(object_poses_inside, device="cuda:0")
        object_poses_outside = to_torch_pose(object_poses_outside, device="cuda:0")

        valid_indices_inside = self.filter_object_poses_in_shelf(object_poses_inside)
        valid_indices_outside = self.filter_object_poses_in_shelf(object_poses_outside)

        if len(valid_indices_inside) == 0 and len(valid_indices_outside) == 0:
            print("No valid poses found")
            return False

        n_inside_limit = min(800, len(valid_indices_inside))
        n_outside_limit = min(800, len(valid_indices_outside))

        poses_inside = poses_inside[valid_indices_inside[:n_inside_limit]]
        poses_outside = poses_outside[valid_indices_outside[:n_outside_limit]]

        object_poses_inside_filtered = object_poses_inside[valid_indices_inside[:n_inside_limit]]
        object_poses_outside_filtered = object_poses_outside[valid_indices_outside[:n_outside_limit]]

        poses_inside_curobo = torch.tensor(poses_inside, device="cuda:0", dtype=torch.float32)
        poses_outside_curobo = torch.tensor(poses_outside, device="cuda:0", dtype=torch.float32)
        poses_inside_curobo[..., 0] = poses_inside_curobo[..., 0] + 0.615
        poses_outside_curobo[..., 0] = poses_outside_curobo[..., 0] + 0.615

        place_joints_inside, feasible_indices_inside = self.action_primitives.inverse_kinematics(poses_inside_curobo)
        place_joints_outside, feasible_indices_outside = self.action_primitives.inverse_kinematics(poses_outside_curobo)

        feasible_mask_inside = feasible_indices_inside[:, 0]
        feasible_mask_outside = feasible_indices_outside[:, 0]

        place_joints_inside = place_joints_inside[feasible_mask_inside][:self.num_envs]
        place_joints_outside = place_joints_outside[feasible_mask_outside][:self.num_envs]

        poses_inside = poses_inside[feasible_mask_inside.cpu().numpy()][:self.num_envs]
        poses_outside = poses_outside[feasible_mask_outside.cpu().numpy()][:self.num_envs]

        object_poses_inside_filtered = object_poses_inside_filtered[feasible_mask_inside][:self.num_envs]
        object_poses_outside_filtered = object_poses_outside_filtered[feasible_mask_outside][:self.num_envs]

        #  ======================== OUTSIDE POSES ======================== #
        self.set_joint_state(place_joints_outside)
        self.set_object_pose(object_name, object_poses_outside_filtered)
        if self.render_mode is not None:
            self.render()
        self.wait(steps=50)

        object_still_in_shelf_indices = []
        for i in range(len(place_joints_outside)):
            if self.check_object_inside_shelf(object_name, env_idx = i) and not self.check_object_within_gripper(object_name, env_idx = i):
                object_still_in_shelf_indices.append(i)

        if len(object_still_in_shelf_indices) == 0:
            print("No objects still in shelf")
            return False

        valid_outside_poses = poses_outside[object_still_in_shelf_indices]

        #  ======================== INSIDE POSES ======================== #
        self.set_joint_state(place_joints_inside)
        self.set_object_pose(object_name, object_poses_inside_filtered)
        if self.render_mode is not None:
            self.render()
        self.wait(steps=50)

        object_still_in_shelf_indices = []
        for i in range(len(place_joints_inside)):
            if self.check_object_inside_shelf(object_name, env_idx = i) and not self.check_object_within_gripper(object_name, env_idx = i):
                object_still_in_shelf_indices.append(i)

        if len(object_still_in_shelf_indices) == 0:
            print("No objects still in shelf")
            return False

        valid_inside_poses = poses_inside[object_still_in_shelf_indices]

        num_poses_to_take = min(len(valid_outside_poses), len(valid_inside_poses), batch_size // 2)

        valid_outside_poses = valid_outside_poses[:num_poses_to_take]
        valid_inside_poses = valid_inside_poses[:num_poses_to_take]

        all_target_poses = np.concatenate([valid_outside_poses, valid_inside_poses], axis=0)
        num_poses = len(all_target_poses)

        object_grasped_poses = self.sample_object_grasped_poses(object_name, n_poses=num_poses)
        self.set_object_pose(object_name, object_grasped_poses)
        if self.render_mode is not None:
            self.render()

        shelf_pcd = self.get_sampled_actor_point_cloud(self.shelf, num_points = num_points // 2)

        object_pcds = []
        for i in range(num_poses):
            object_pcd = self.get_sampled_actor_point_cloud(self.objects[object_name], env_idx = i, num_points = num_points // 2)
            object_pcds.append(object_pcd)

        shelf_pcd_reshaped = shelf_pcd[np.newaxis, :, :]
        repeated_shelf_pcd = np.repeat(shelf_pcd_reshaped, num_poses, axis=0)

        stacked_object_pcd = np.stack(object_pcds, axis=0)

        combined_pcd = np.concatenate([repeated_shelf_pcd, stacked_object_pcd], axis=1)

        object_grasped_transformations = pose_to_transformation(object_grasped_poses.cpu().numpy(), format='wxyz')
        starting_grasp_transformation = object_grasped_transformations @ grasp_transformation[np.newaxis, :, :]
        starting_grasp_poses = transformation_to_pose(starting_grasp_transformation, format='wxyz')

        return starting_grasp_poses, combined_pcd, all_target_poses

    def test_pushing(self):
        """Smoke-test Push on the first object, then pushes started from feasible grasp poses.

        Legacy: the push and grasp samplers it calls now live on ActionSampler.
        """

        push_start_poses, \
        push_directions, \
        push_distances = self.sample_push_actions(
            object_name=self.object_names[0],
            n=1000,
            min_push_distance=0.12,
            max_push_distance=0.22,
            max_offset_angle=0.,
            env_idx=0
        )

        z_values = push_start_poses[:, 2]
        sort_indices = torch.argsort(z_values)
        push_start_poses = push_start_poses[sort_indices]
        push_directions = push_directions[sort_indices]
        push_distances = push_distances[sort_indices]

        self.action_primitives.CloseGripper()

        self.action_primitives.Push(
            push_start=push_start_poses[:self.num_envs],
            push_direction=push_directions[:self.num_envs],
            push_distance=push_distances[:self.num_envs],
        )

        grasp_poses = []
        grasp_costs = []
        grasp_object_ids = []

        for object_index in range(len(self.object_names)):
            object_name = self.object_names[object_index]
            obj_grasp_poses = self.get_object_grasp_poses(
                object_name,
                env_idx=0,
                grasp_poses=self.grasps[object_name]
            )
            grasp_poses.append(torch.tensor(obj_grasp_poses, device="cuda:0", dtype=torch.float32))
            grasp_costs.append(torch.tensor(self.grasp_costs[object_name], device="cuda:0", dtype=torch.float32))
            grasp_object_ids.append(torch.full((obj_grasp_poses.shape[0],), object_index, device="cuda:0", dtype=torch.int64))

        grasp_poses = torch.cat(grasp_poses, dim=0)
        grasp_costs = torch.cat(grasp_costs, dim=0)
        grasp_object_ids = torch.cat(grasp_object_ids, dim=0)

        print(f"Doing IK for grasps")
        curobo_grasp_poses = grasp_poses.clone()
        curobo_grasp_poses[..., 0] = curobo_grasp_poses[..., 0] + 0.615
        grasp_joints, feasible_indices = self.action_primitives.inverse_kinematics(curobo_grasp_poses)

        grasp_poses = grasp_poses[torch.where(feasible_indices[:, 0])[0]]

        push_start_poses = move_pose_along_local_y(grasp_poses[:self.num_envs], 0.15)
        push_start_poses = move_pose_along_local_z(push_start_poses, 0.08)
        push_start_poses = to_torch_pose(push_start_poses)
        push_direction = torch.tensor([1], device="cuda:0", dtype=torch.int64).repeat(self.num_envs)
        push_distance = torch.tensor([-0.2], device="cuda:0", dtype=torch.float32).repeat(self.num_envs)

        self.action_primitives.CloseGripper()

        self.action_primitives.Push(
            push_start=push_start_poses,
            push_direction=push_direction,
            push_distance=push_distance,
        )

    def generate_plans(self):
        """Pick a grasp and a shelf placement for each object in turn.

        Legacy: relies on helpers that are no longer defined.

        Returns:
            Stacked grasp poses, object poses, placement poses, grasp joints,
            placement joints, and the object order.
        """
        self.update_motion_planner()
        self.set_gripper_friction(friction_coefficient=5.0)

        print("Sampling shelf poses")
        shelf_poses, shelf_joints = self.sample_gripper_poses_in_shelf(
            n=self.num_envs,
            depth_tolerance=0.05,
            width_tolerance=0.05,
            height_tolerance=0.05
        )

        available_object_ids = list(np.arange(len(self.object_names)))

        grasp_poses = []
        placement_poses = []
        grasp_joints = []
        placement_joints = []
        object_order = []
        object_poses = []

        while len(available_object_ids) > 0:

            object_id, \
            object_pose, \
            grasp_pose, \
            grasp_joint_config, \
            placement_pose, \
            placement_joint_config = self.get_grasp_and_placement_pose(available_object_ids, shelf_poses)

            grasp_poses.append(grasp_pose)
            placement_poses.append(placement_pose)
            grasp_joints.append(grasp_joint_config)
            placement_joints.append(placement_joint_config)
            object_order.append(object_id)
            object_poses.append(object_pose)

            available_object_ids.remove(object_id)

        grasp_poses = torch.stack(grasp_poses, dim=0)
        placement_poses = torch.stack(placement_poses, dim=0)
        grasp_joints = torch.stack(grasp_joints, dim=0)
        placement_joints = torch.stack(placement_joints, dim=0)
        object_poses = torch.stack(object_poses, dim=0)
        object_order = torch.tensor(object_order, device="cuda:0", dtype=torch.int64)

        return grasp_poses, object_poses, placement_poses, grasp_joints, placement_joints, object_order

    ###########################################################################

    def save_environment(self, save_dir, env_id: int = 0):
        """Save slot 0's object poses, shelf parameters, and URDF/mesh files as env_<id>/.

        Args:
            save_dir: Parent directory for the env folder.
            env_id: Index used in the folder name.

        Returns:
            Path to the written env folder (holds environment.yaml).
        """
        env_folder_name = f"env_{env_id}"
        env_path = os.path.join(save_dir, env_folder_name)
        os.makedirs(env_path, exist_ok=True)

        print(f"Saving environment to: {env_path}")

        object_info = {}
        for i, object_name in enumerate(self.object_names):
            current_pose = self.get_object_pose(object_name)[0].cpu().numpy()

            urdf_path = None
            if hasattr(self, 'urdf_paths') and object_name in self.urdf_paths:
                urdf_path = self.urdf_paths[object_name]

            bounds = self.bounds.get(object_name)
            quat = self.object_quaternions.get(object_name)
            scale_factor = self.object_scale_factors.get(object_name)

            if bounds is not None:
                bounds = [[float(b) for b in bound] for bound in bounds]

            if quat is not None:
                if isinstance(quat, np.ndarray):
                    quat = quat.tolist()
                else:
                    quat = [float(q) for q in quat]

            object_info[f'object_{i}'] = {
                'object_id': str(object_name),
                'pose': current_pose.tolist(),
                'urdf_path': urdf_path,
                'bounds': bounds,
                'quaternion': quat,
                'scale_factor': float(scale_factor) if scale_factor is not None else None
            }

            if urdf_path and os.path.exists(urdf_path):
                object_urdf_name = f"object_{i}_{object_name}.urdf"
                dest_urdf_path = os.path.join(env_path, object_urdf_name)
                shutil.copy2(urdf_path, dest_urdf_path)
                object_info[f'object_{i}']['urdf_filename'] = object_urdf_name
            else:
                raise ValueError(f"URDF path for {object_name} is not found")

        shelf_pose = self.get_object_pose(self.shelf)[0].cpu().numpy() if hasattr(self, 'shelf') else self.shelf_pose

        shelf_info = {
            'width': float(self.shelf_width),
            'height': float(self.shelf_height),
            'depth': float(self.shelf_depth),
            'wall_thickness': float(self.wall_thickness),
            'distance_from_floor': float(self.shelf_distance_from_floor),
            'bottom_wall_height': float(self.bottom_wall_height) if hasattr(self, 'bottom_wall_height') else None,
            'pose': shelf_pose.tolist()
        }

        shelf_folder = "assets/shelf"
        shelf_obj_files = ["top_wall.obj", "vertical_wall.obj", "back_wall.obj", "bottom_wall.obj"]
        shelf_urdf_source = os.path.join(shelf_folder, "shelf.urdf")

        if os.path.exists(shelf_urdf_source):
            shelf_urdf_dest = os.path.join(env_path, "shelf.urdf")
            shutil.copy2(shelf_urdf_source, shelf_urdf_dest)
            shelf_info['urdf_filename'] = "shelf.urdf"

            for obj_file in shelf_obj_files:
                obj_source = os.path.join(shelf_folder, obj_file)
                if os.path.exists(obj_source):
                    obj_dest = os.path.join(env_path, obj_file)
                    shutil.copy2(obj_source, obj_dest)
                else:
                    raise ValueError(f"Shelf OBJ file {obj_file} not found at {obj_source}")
        else:
            raise ValueError(f"Shelf URDF not found at {shelf_urdf_source}")

        env_data = {
            'shelf': shelf_info,
            'objects': object_info,
        }

        def convert_numpy_types(obj):
            """Recursively turn numpy/torch values into plain Python types for YAML.

            Args:
                obj: Nested dict/list/array/tensor/scalar.
            """
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, (np.str_, np.unicode_)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        env_data = convert_numpy_types(env_data)

        yaml_path = os.path.join(env_path, "environment.yaml")
        OmegaConf.save(env_data, yaml_path)

        print(f"Environment saved successfully to {env_path}")

        return env_path

    def infer_q_function(self, parent_state, grasp_actions, place_actions):
        """Predict Q values for (grasp, place) pairs at one state; restores the sim state after.

        Args:
            parent_state: (state_dim,) sim state before the grasp.
            grasp_actions: (N, >=7) grasp poses (xyz + wxyz quat), world frame.
            place_actions: (N, >=7) matching place poses, world frame.

        Returns:
            (N,) tensor of predicted success values.
        """
        assert len(grasp_actions) == len(place_actions), (
            "grasp_actions and place_actions must have the same length"
        )
        num_actions = len(grasp_actions)

        saved_state = self.get_state().clone()

        self.set_state(parent_state.unsqueeze(0).repeat(self.num_envs, 1))
        pcd = self.get_sampled_point_cloud(env_idx=0)
        pcd = downsample_point_cloud(pcd, num_points=4096)
        input_pcd = torch.tensor(pcd, device="cuda:0", dtype=torch.float32)
        input_pcd[..., 0] = input_pcd[..., 0] + 0.615
        input_pcd = input_pcd.unsqueeze(0).expand(num_actions, -1, -1)

        # Build 8D grasp/place tokens with correct gripper convention:
        # grasp ends with gripper CLOSED = 1.0, placement ends with gripper OPEN = 0.0.
        grasp_poses = grasp_actions[..., :7].clone().to("cuda:0", dtype=torch.float32)
        grasp_poses[..., 0] = grasp_poses[..., 0] + 0.615
        grasp_gripper = torch.ones(num_actions, 1, device="cuda:0", dtype=torch.float32)
        grasp_token = torch.cat([grasp_poses, grasp_gripper], dim=-1)  # (N, 8)

        place_poses = place_actions[..., :7].clone().to("cuda:0", dtype=torch.float32)
        place_poses[..., 0] = place_poses[..., 0] + 0.615
        place_gripper = torch.zeros(num_actions, 1, device="cuda:0", dtype=torch.float32)
        place_token = torch.cat([place_poses, place_gripper], dim=-1)  # (N, 8)

        # (N, 2, 8)
        action_input = torch.stack([grasp_token, place_token], dim=1)

        max_batch_size = 4
        all_values = []
        with torch.no_grad():
            for i in range(0, num_actions, max_batch_size):
                end_idx = min(i + max_batch_size, num_actions)
                value_chunk = self.value_model(
                    pcd=input_pcd[i:end_idx],
                    actions=action_input[i:end_idx],
                    targets=None,
                )
                all_values.append(value_chunk.squeeze(-1))

        q_values = torch.cat(all_values, dim=0)
        self.set_state(saved_state)
        return q_values

    def all_objects_packed(self, env_idx: int = 0) -> bool:
        """Return True if every object's volume is in the shelf in one env slot.

        Args:
            env_idx: Env slot to check.
        """
        rew = self.compute_dense_reward()[env_idx].item()
        return rew >= 1.0 - 1e-5

    def infer_value_at_state(self, state: torch.Tensor) -> float:
        """Predict the value of one state, querying with its current gripper pose and opening.

        Args:
            state: (state_dim,) sim state.

        Returns:
            Predicted value, or 0.0 if no value network is loaded.
        """
        if getattr(self, "value_model", None) is None:
            return 0.0

        from disco.utils.conversion_utils import (
            downsample_point_cloud,
        )

        saved_state = self.get_state().clone()
        self.set_state(state.unsqueeze(0).repeat(self.num_envs, 1))

        pcd = self.get_sampled_point_cloud(env_idx=0)
        pcd = downsample_point_cloud(pcd, num_points=4096)
        input_pcd = torch.tensor(pcd, device="cuda:0", dtype=torch.float32)
        input_pcd[..., 0] = input_pcd[..., 0] + 0.615
        input_pcd = input_pcd.unsqueeze(0)

        gp = self.get_gripper_pose()[0].to("cuda:0", dtype=torch.float32)
        goal_pose = gp.clone()
        goal_pose[0] = goal_pose[0] + 0.615

        gq = self.get_gripper_state()[0].to("cuda:0", dtype=torch.float32)
        gripper_scalar = (gq.mean().clamp(0.0, 0.04) / 0.04).clamp(0.0, 1.0)

        action = torch.cat([goal_pose, gripper_scalar.unsqueeze(0)], dim=-1).unsqueeze(0)

        self.value_model.eval()
        with torch.no_grad():
            v = self.value_model(pcd=input_pcd, actions=action, targets=None)
        val = float(v.squeeze().item())

        self.set_state(saved_state)
        return val

    def evaluate(self):
        """Return a placeholder info dict with success all False."""
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def compute_dense_reward(self, obs=None, action=None, info=None):
        """Return the fraction of total object volume inside the shelf, per env slot.

        Args:
            obs: Unused.
            action: Unused.
            info: Unused.

        Returns:
            (num_envs,) tensor in [0, 1].
        """
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for env_idx in range(self.num_envs):
            _, object_indices_in_shelf = self.number_of_objects_in_shelf(env_idx=env_idx)
            packed_volume = sum(self.object_volumes[self.object_names[i]] for i in object_indices_in_shelf)
            rewards[env_idx] = packed_volume / self.total_volume

        return rewards

    def compute_normalized_dense_reward(self, obs=None, action=None, info=None):
        """Return a placeholder constant reward of 0.5 per env slot.

        Args:
            obs: Unused.
            action: Unused.
            info: Unused.
        """
        return torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 0.5
