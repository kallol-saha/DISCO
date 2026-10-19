"""
Shelf packing environment loaded from a saved environment.yaml.
"""
import os
import numpy as np
import sapien
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from mani_skill.utils.registration import register_env

from disco.envs.shelf_packing_base import ShelfPackingBase
from disco.utils.common import (
    load_acronym_object_and_grasps,
    to_sapien_pose,
)


@register_env("ShelfPacking-v1", max_episode_steps=10000)
class ShelfPacking(ShelfPackingBase):
    """Shelf-packing env loaded from a saved env_<i>/environment.yaml scene."""

    def __init__(
        self,
        *args,
        env_path="assets/environments/shelf_packing/train",
        env_index=0,
        **kwargs):
        """Read the scene file, then build the env.

        Args:
            *args: Forwarded to ShelfPackingBase.
            env_path: Directory holding the env_<i>/ scene folders.
            env_index: Which env_<i> folder to load.
            **kwargs: Forwarded to ShelfPackingBase (e.g. num_envs).
        """

        self.env_path = env_path
        self.env_index = env_index

        # Load replay data before super().__init__() because _initialize_scene()
        # (called from _load_scene during super().__init__()) needs self.scene_data
        self._load_replay_data()

        super().__init__(*args, **kwargs)

    def _load_replay_data(self):
        """Load env_<env_index>/environment.yaml into `self.scene_data`."""
        env_yaml_path = os.path.join(self.env_path, f"env_{self.env_index}", "environment.yaml")

        if not os.path.exists(env_yaml_path):
            raise FileNotFoundError(f"environment.yaml not found in {self.env_path}/env_{self.env_index}")

        self.scene_data = OmegaConf.load(env_yaml_path)

    def _initialize_scene(self):
        """Build the objects (with their grasps and volumes) and the shelf from the scene file."""

        self.object_names = []

        object_data = self.scene_data.objects
        self.num_objects = len(object_data)

        grasp_dir = "assets/final_grasps"

        for i, (_, object_properties) in enumerate(object_data.items()):

            object_name = object_properties.object_id
            object_pose = np.array(object_properties.pose)

            # Load URDF path, bounds, quat, and scale_factor from YAML
            urdf_path = os.path.join(self.env_path, f"env_{self.env_index}", object_properties.urdf_filename)
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"Object URDF not found at {urdf_path}")

            # Load bounds, quaternion, and scale_factor from YAML
            self.bounds[object_name] = np.array(object_properties.bounds)
            self.object_quaternions[object_name] = np.array(object_properties.quaternion)
            self.object_scale_factors[object_name] = float(object_properties.scale_factor)

            # Only load grasps and grasp_costs from object directory (load_full=False)
            object_dir = os.path.join("assets/acronym_objects", object_name)
            grasps, grasp_costs, object_volume = load_acronym_object_and_grasps(object_dir, grasp_dir, load_full=False)
            self.grasps[object_name] = grasps
            self.grasp_costs[object_name] = grasp_costs
            self.object_volumes[object_name] = object_volume
            self.object_poses[object_name] = object_pose
            self.object_names.append(object_name)

            actor_builders = self.urdf_loader.parse(urdf_path)["actor_builders"]
            builder = actor_builders[0]
            builder.initial_pose = to_sapien_pose(object_pose)
            self.objects[object_name] = builder.build(name=object_name)
            self.object_ids[object_name] = self.objects[object_name].per_scene_id[0].item()
            self.actors.append(self.objects[object_name])
            self.movable_actors.append(self.objects[object_name])

        # ----------------- SHELF ----------------- #

        shelf_data = self.scene_data.shelf

        # Dimensions:
        self.shelf_width = shelf_data.width
        self.shelf_depth = shelf_data.depth
        self.shelf_height = shelf_data.height
        self.shelf_distance_from_floor = shelf_data.distance_from_floor
        self.wall_thickness = shelf_data.wall_thickness
        self.bottom_wall_height = shelf_data.bottom_wall_height

        self.shelf_ceiling = self.shelf_distance_from_floor + self.wall_thickness + self.shelf_height

        shelf_urdf_path = os.path.join(self.env_path, f"env_{self.env_index}", shelf_data.urdf_filename)
        if not os.path.exists(shelf_urdf_path):
            raise FileNotFoundError(f"Shelf URDF not found at {shelf_urdf_path}")

        # Pose:
        self.shelf_pose = np.array(shelf_data.pose)

        # Extract shelf rotation:
        quat = np.array(shelf_data.pose[3:])
        rotation = R.from_quat(quat, scalar_first=True)
        self.shelf_rotation = rotation.as_euler('xyz')[2]

        # Extract shelf position:
        self.shelf_position = np.array(shelf_data.pose[:3])

        # Load in the shelf:
        actor_builders = self.urdf_loader.parse(shelf_urdf_path)["articulation_builders"]
        builder = actor_builders[0]
        builder.initial_pose = to_sapien_pose(self.shelf_pose)
        self.shelf = builder.build(name="shelf")
        self.actors.append(self.shelf)

        self.total_volume = sum(self.object_volumes.values())


# Backward compatibility alias
ShelfPackingMultiObject = ShelfPacking
