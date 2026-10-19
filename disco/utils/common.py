"""
ACRONYM object and grasp loading, URDF generation and placement orientations.
"""
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import sapien
from mani_skill.envs.scene import ManiSkillScene
from transforms3d import quaternions
from transforms3d import axangles
from typing import Union
import torch
from scipy.spatial.transform import Rotation as R
import trimesh
import h5py
import matplotlib.pyplot as plt

from disco.utils.conversion_utils import (
    transformation_to_pose,
    pose_to_transformation,
    invert_transformation,
    transform_pcd,
    move_pose_along_local_z,
)
from disco.utils.visualization.point_cloud_structures import make_gripper_visualization
from disco.utils.visualization.plotting import plot_pcd, visualize_poses_in_pointcloud


def object_hash(name: str) -> str:
    """Return the bare object hash from a name, e.g. 'object_abc123_4' -> 'abc123'.

    Args:
        name: Object name, optionally prefixed with 'object_' and suffixed with '_<index>'.
    """
    return name.rsplit("_", 1)[0].replace("object_", "", 1)


def load_grasps(grasps_file_path):
    """Load grasps from a YAML file and add the 180-degree z-flipped copy of each.

    Each grasp is also turned 90 degrees about its z-axis and pushed 8 cm along it.

    Args:
        grasps_file_path: Path to the grasp YAML file.

    Returns:
        (2N, 7) grasp poses (x, y, z, qw, qx, qy, qz) and their (2N,) confidences.
    """

    # Load grasps from json file
    if os.path.exists(grasps_file_path):
        with open(grasps_file_path, 'r') as f:
            grasps = OmegaConf.load(f)
        print(f"Loaded {len(grasps)} grasps from {grasps_file_path}")
    else:
        print(f"No grasp file found at {grasps_file_path}")
        grasps = []

    grasp_poses = grasps["grasps"]
    num_grasps = len(grasp_poses)

    # TODO: Instead of loading in grasps in a for loop like this, store a big numpy array of grasps for each
    grasps = np.zeros((num_grasps * 2, 7)) # --> (x, y, z, qw, qx, qy, qz)
    confidences = np.zeros((num_grasps * 2,))

    for i in tqdm(range(num_grasps)):
        grasp_pose = grasp_poses[f"grasp_{i}"]
        pos = np.array(grasp_pose["position"])
        quat = np.append(
            grasp_pose["orientation"]["w"],
            np.array(grasp_pose["orientation"]["xyz"])
            )   # This is in wxyz
        # Rotate grasp orientation by 90 degrees about z and convert back to quaternion (wxyz)
        rot_z = axangles.axangle2mat([0, 0, 1], np.deg2rad(90.0))
        rot_base = quaternions.quat2mat(quat)  # expects wxyz
        rot = np.matmul(rot_base, rot_z)
        z_offset = 0.08 * rot[:, 2]  # Extract rotated z-axis and scale
        pos = pos + z_offset
        quat_rotated = quaternions.mat2quat(rot)  # returns wxyz

        grasps[i, :3] = pos
        grasps[i, 3:] = quat_rotated
        confidences[i] = grasp_pose["confidence"]

        # Rotate the grasp pose by 180 degrees about the z-axis
        quat_rotated_180 = flip_quat_about_z(quat_rotated)

        grasps[i + num_grasps, :3] = pos
        grasps[i + num_grasps, 3:] = quat_rotated_180
        confidences[i + num_grasps] = grasp_pose["confidence"]

    return grasps, confidences

def load_acronym_object_and_grasps(object_dir, grasp_dir, mass = 0.5, inertia = 0.001, load_full = True):
    """Load an ACRONYM object's grasps sorted by cost, and optionally build its URDF.

    Args:
        object_dir: ACRONYM object folder (mesh + annotation.h5).
        grasp_dir: Folder of preprocessed per-object grasp .npz files.
        mass: Object mass written to the URDF (kg).
        inertia: Diagonal inertia written to the URDF.
        load_full: Also write the URDF and sample a random yaw; else return grasps only.

    Returns:
        If load_full: (urdf_path, grasps, bounds, quat_wxyz, scale_factor, grasp_costs, volume).
        Otherwise: (grasps, grasp_costs, volume). Grasps are (N, 7) wxyz poses, lowest cost first.
    """

    # Load pre-processed grasps from grasp_dir
    object_id = os.path.basename(object_dir)
    grasp_npz_path = os.path.join(grasp_dir, f"{object_id}.npz")
    grasp_data = np.load(grasp_npz_path)

    grasp_annotation_file = os.path.join(object_dir, 'annotation.h5')
    grasp_annotation = h5py.File(grasp_annotation_file, 'r')

    object_volume = grasp_annotation['object/volume'][()]

    grasps = grasp_data['grasps']
    closing_angular = grasp_data['closing_angular']
    closing_linear = grasp_data['closing_linear']
    shaking_angular = grasp_data['shaking_angular']
    shaking_linear = grasp_data['shaking_linear']
    scale_factor = float(grasp_data['scale_factor'])

    # Normalize quality metrics and compute grasp costs
    arrays = [closing_angular, closing_linear, shaking_angular, shaking_linear]
    normalized_arrays = []
    for arr in arrays:
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            normalized = (arr - arr_min) / (arr_max - arr_min)
        else:
            normalized = np.zeros_like(arr)
        normalized_arrays.append(normalized)

    grasp_costs = sum(normalized_arrays)

    # Rank grasps based on costs (lower is better)
    grasp_rankings = np.argsort(grasp_costs)
    output_grasps = grasps[grasp_rankings]
    grasp_costs = grasp_costs[grasp_rankings]

    # If only loading grasps and costs, return early
    if not load_full:
        return output_grasps, grasp_costs, object_volume

    # Full loading: generate URDF, bounds, and quaternion
    mesh_path = os.path.join(object_dir, 'model_repositioned_vhacd.obj')
    
    # Load mesh for bounds calculation
    mesh = trimesh.load(mesh_path)
    verts = mesh.vertices

    # Generate random color
    random_color = np.random.rand(3)
    color_rgba = f"{random_color[0]:.3f} {random_color[1]:.3f} {random_color[2]:.3f} 1.0"

    # Create URDF content
    urdf_content = f'''<?xml version="1.0"?>
<robot name="dummy">
  <material name="random_material">
    <color rgba="{color_rgba}"/>
  </material>
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{inertia}" ixy="0" ixz="0" iyy="{inertia}" iyz="0" izz="{inertia}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{scale_factor} {scale_factor} {scale_factor}"/>
      </geometry>
      <material name="random_material"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{scale_factor} {scale_factor} {scale_factor}"/>
      </geometry>
    </collision>
  </link>
</robot>'''

    # Save URDF
    urdf_path = f"assets/object_urdfs/object_{object_id}.urdf"
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)

    # Random z-axis rotation
    z_angle = np.random.uniform(0, 360)
    rotation = R.from_euler('z', z_angle, degrees=True)

    # Apply rotation to scaled vertices to get new bounds
    rotated_verts = rotation.apply(verts * scale_factor)

    # Calculate new bounds
    new_bounds = [
        [np.min(rotated_verts[:, 0]), np.max(rotated_verts[:, 0])],  # x bounds
        [np.min(rotated_verts[:, 1]), np.max(rotated_verts[:, 1])],  # y bounds
        [np.min(rotated_verts[:, 2]), np.max(rotated_verts[:, 2])]   # z bounds
    ]

    # Get quaternion (w, x, y, z format)
    quaternion = rotation.as_quat()  # Returns [x, y, z, w]
    quaternion_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])  # Convert to [w, x, y, z]

    return urdf_path, output_grasps, new_bounds, quaternion_wxyz, scale_factor, grasp_costs, object_volume

def filter_object_save_grasps(
    object_dir,
    lower_dim_bound: int = 0.03,
    upper_dim_bound: int = 0.15,
    closing_angular_bound: int = 0.05,
    closing_linear_bound: int = 0.003,
    shaking_angular_bound: int = 0.05,
    shaking_linear_bound: int = 0.003,
    min_valid_grasps: int = 5,
    ):
    """Filter an ACRONYM object by size and keep its grasps that held stably in simulation.

    Args:
        object_dir: ACRONYM object folder (mesh, grasp .npy, annotation.h5).
        lower_dim_bound: Minimum allowed smallest object dimension (m).
        upper_dim_bound: Maximum allowed largest object dimension (m).
        closing_angular_bound: Max object rotation while the gripper closes.
        closing_linear_bound: Max object translation while the gripper closes.
        shaking_angular_bound: Max object rotation while shaking.
        shaking_linear_bound: Max object translation while shaking.
        min_valid_grasps: Reject the object if fewer grasps survive.

    Returns:
        Dict of (N, 7) wxyz grasps, their quality metrics and scale_factor, or None if rejected.
    """
    # mesh_path = os.path.join(object_dir, 'model_repositioned_vhacd.obj')
    # grasp_file = os.path.join(object_dir, 'annotation_rescaled_repositioned.npy')
    # grasp_annotation_file = os.path.join(object_dir, 'annotation.h5')

    mesh_path = os.path.join(object_dir, 'model_repositioned_vhacd.obj')
    grasp_file = os.path.join(object_dir, 'annotation_rescaled_repositioned.npy')
    grasp_annotation_file = os.path.join(object_dir, 'annotation.h5')

    mesh = trimesh.load(mesh_path)
    grasps = np.load(grasp_file)

    # NOTE: Uncomment this if you want to use the scale factor from the grasp annotation file.
    grasp_annotation = h5py.File(grasp_annotation_file, 'r')
    scale_factor = grasp_annotation['object/scale'][()]

    verts = mesh.vertices
    verts = verts * scale_factor
    grasps[:, :3, 3] = grasps[:, :3, 3] * scale_factor

    # Find the largest dimension in x and y axes
    x_span = np.max(verts[:, 0]) - np.min(verts[:, 0])
    y_span = np.max(verts[:, 1]) - np.min(verts[:, 1])
    z_span = np.max(verts[:, 2]) - np.min(verts[:, 2])

    largest_dimension = max(x_span, y_span, z_span)
    lowest_dimension = min(x_span, y_span, z_span)
    lowest_z = np.min(verts[:, 2])

    within_size_limits = (lowest_dimension > lower_dim_bound) and (lowest_dimension < 0.06) and (largest_dimension < upper_dim_bound)
    if not within_size_limits:
        return None

    # Filtering criteria:
    object_in_gripper = grasp_annotation['grasps/qualities/flex/object_in_gripper'][()]
    object_motion_during_closing_angular = grasp_annotation['grasps/qualities/flex/object_motion_during_closing_angular'][()]
    object_motion_during_closing_linear = grasp_annotation['grasps/qualities/flex/object_motion_during_closing_linear'][()]
    object_motion_during_shaking_angular = grasp_annotation['grasps/qualities/flex/object_motion_during_shaking_angular'][()]
    object_motion_during_shaking_linear = grasp_annotation['grasps/qualities/flex/object_motion_during_shaking_linear'][()]

    object_in_gripper_mask = (object_in_gripper == 1)

    # Filter grasps and metrics by object_in_gripper
    grasps_in_gripper = grasps[object_in_gripper_mask]

    closing_angular = object_motion_during_closing_angular[object_in_gripper_mask]
    closing_linear = object_motion_during_closing_linear[object_in_gripper_mask]
    shaking_angular = object_motion_during_shaking_angular[object_in_gripper_mask]
    shaking_linear = object_motion_during_shaking_linear[object_in_gripper_mask]

    closing_angular_validity = closing_angular < closing_angular_bound
    closing_linear_validity = closing_linear < closing_linear_bound
    shaking_angular_validity = shaking_angular < shaking_angular_bound
    shaking_linear_validity = shaking_linear < shaking_linear_bound
    grasp_above_table = (grasps_in_gripper[:, 2, 3] > lowest_z) & ((grasps_in_gripper[:, 2, 3] - lowest_z) > 0.04)

    validity_mask = closing_angular_validity & closing_linear_validity & shaking_angular_validity & shaking_linear_validity & grasp_above_table
        
    if validity_mask.sum() < min_valid_grasps:
        return None

    # Filter to only valid grasps
    valid_grasps = grasps_in_gripper[validity_mask]
    valid_closing_angular = closing_angular[validity_mask]
    valid_closing_linear = closing_linear[validity_mask]
    valid_shaking_angular = shaking_angular[validity_mask]
    valid_shaking_linear = shaking_linear[validity_mask]

    # Process grasps: scale positions and convert to pose format
    scaled_grasps = valid_grasps.copy()     # NOTE: I scaled it above, so this variable name is misleading, I just didn't change the name but have to.

    # Convert transformation matrices to pose format (x, y, z, qw, qx, qy, qz)
    scaled_grasp_poses = transformation_to_pose(scaled_grasps, format='wxyz')

    # Apply 90-degree z-rotation to each grasp
    num_grasps = scaled_grasp_poses.shape[0]
    output_grasps = np.zeros((num_grasps, 7))  # (x, y, z, qw, qx, qy, qz)

    for i in range(num_grasps):
        grasp_quat = scaled_grasp_poses[i, 3:]
        grasp_pos = scaled_grasp_poses[i, :3]

        rot_z = axangles.axangle2mat([0, 0, 1], np.deg2rad(90.0))
        rot_base = quaternions.quat2mat(grasp_quat)  # expects wxyz
        rot = np.matmul(rot_base, rot_z)
        grasp_quat_rotated = quaternions.mat2quat(rot)  # returns wxyz

        output_grasps[i, :3] = grasp_pos
        output_grasps[i, 3:] = grasp_quat_rotated

    return {
        'grasps': output_grasps,
        'closing_angular': valid_closing_angular,
        'closing_linear': valid_closing_linear,
        'shaking_angular': valid_shaking_angular,
        'shaking_linear': valid_shaking_linear,
        'scale_factor': scale_factor,
    }

def filter_object_save_grasps_v2(
    object_dir,
    lower_dim_bound: int = 0.03,
    upper_dim_bound: int = 0.15,
    closing_angular_bound: int = 0.05,
    closing_linear_bound: int = 0.003,
    shaking_angular_bound: int = 0.05,
    shaking_linear_bound: int = 0.003,
    min_valid_grasps: int = 5,
    gripper_z_offset: float = 0.105,
    ):
    """Filter an ACRONYM object and its grasps like v1, also dropping grasps of bad finger width.

    Args:
        object_dir: ACRONYM object folder (mesh, grasp .npy, annotation.h5).
        lower_dim_bound: Minimum allowed smallest object dimension (m).
        upper_dim_bound: Maximum allowed largest object dimension (m).
        closing_angular_bound: Max object rotation while the gripper closes.
        closing_linear_bound: Max object translation while the gripper closes.
        shaking_angular_bound: Max object rotation while shaking.
        shaking_linear_bound: Max object translation while shaking.
        min_valid_grasps: Reject the object if fewer grasps survive.
        gripper_z_offset: Unused.

    Returns:
        Dict of (N, 7) wxyz grasps, their quality metrics and scale_factor, or None if rejected.
    """
    mesh_path = os.path.join(object_dir, 'model_repositioned_vhacd.obj')
    grasp_file = os.path.join(object_dir, 'annotation_rescaled_repositioned.npy')
    grasp_annotation_file = os.path.join(object_dir, 'annotation.h5')

    mesh = trimesh.load(mesh_path)
    grasps = np.load(grasp_file)

    grasp_annotation = h5py.File(grasp_annotation_file, 'r')
    scale_factor = grasp_annotation['object/scale'][()]

    # Sample point cloud from mesh surface (not interior) and scale it
    points, _ = trimesh.sample.sample_surface(mesh, 10000)
    points = np.array(points) * scale_factor

    verts = mesh.vertices * scale_factor
    grasps[:, :3, 3] = grasps[:, :3, 3] * scale_factor

    # Find dimensions
    x_span = np.max(verts[:, 0]) - np.min(verts[:, 0])
    y_span = np.max(verts[:, 1]) - np.min(verts[:, 1])
    z_span = np.max(verts[:, 2]) - np.min(verts[:, 2])

    largest_dimension = max(x_span, y_span, z_span)
    lowest_dimension = min(x_span, y_span, z_span)
    lowest_z = np.min(verts[:, 2])

    within_size_limits = (largest_dimension < upper_dim_bound) & (lowest_dimension > lower_dim_bound)
    if not within_size_limits:
        return None

    # Filtering criteria
    object_in_gripper = grasp_annotation['grasps/qualities/flex/object_in_gripper'][()]
    object_motion_during_closing_angular = grasp_annotation['grasps/qualities/flex/object_motion_during_closing_angular'][()]
    object_motion_during_closing_linear = grasp_annotation['grasps/qualities/flex/object_motion_during_closing_linear'][()]
    object_motion_during_shaking_angular = grasp_annotation['grasps/qualities/flex/object_motion_during_shaking_angular'][()]
    object_motion_during_shaking_linear = grasp_annotation['grasps/qualities/flex/object_motion_during_shaking_linear'][()]

    object_in_gripper_mask = (object_in_gripper == 1)

    # Filter grasps and metrics by object_in_gripper
    grasps_in_gripper = grasps[object_in_gripper_mask]

    closing_angular = object_motion_during_closing_angular[object_in_gripper_mask]
    closing_linear = object_motion_during_closing_linear[object_in_gripper_mask]
    shaking_angular = object_motion_during_shaking_angular[object_in_gripper_mask]
    shaking_linear = object_motion_during_shaking_linear[object_in_gripper_mask]

    closing_angular_validity = closing_angular < closing_angular_bound
    closing_linear_validity = closing_linear < closing_linear_bound
    shaking_angular_validity = shaking_angular < shaking_angular_bound
    shaking_linear_validity = shaking_linear < shaking_linear_bound
    grasp_above_table = (grasps_in_gripper[:, 2, 3] > lowest_z) & ((grasps_in_gripper[:, 2, 3] - lowest_z) > 0.04)

    validity_mask = closing_angular_validity & closing_linear_validity & shaking_angular_validity & shaking_linear_validity & grasp_above_table

    # Filter to only valid grasps
    valid_grasps = grasps_in_gripper[validity_mask]
    valid_closing_angular = closing_angular[validity_mask]
    valid_closing_linear = closing_linear[validity_mask]
    valid_shaking_angular = shaking_angular[validity_mask]
    valid_shaking_linear = shaking_linear[validity_mask]

    # Filter out very small or very large grasps:
    valid_grasp_poses = transformation_to_pose(valid_grasps, format='wxyz')
    gripper_width_mask = np.ones(len(valid_grasps), dtype=bool)
    for i in range(valid_grasp_poses.shape[0]):
        grasp_pose = valid_grasp_poses[i]
        gripper_width = compute_gripper_width(grasp_pose, points, visualize=True)
        # print(gripper_width, object_dir)
        if gripper_width < 0.015 or gripper_width > 0.06:
            gripper_width_mask[i] = False

    valid_grasps = valid_grasps[gripper_width_mask]
    valid_closing_angular = valid_closing_angular[gripper_width_mask]
    valid_closing_linear = valid_closing_linear[gripper_width_mask]
    valid_shaking_angular = valid_shaking_angular[gripper_width_mask]
    valid_shaking_linear = valid_shaking_linear[gripper_width_mask]

    if len(valid_grasps) < min_valid_grasps:
        return None

    # Convert transformation matrices to pose format (x, y, z, qw, qx, qy, qz)
    scaled_grasp_poses = transformation_to_pose(valid_grasps, format='wxyz')

    # Apply 90-degree z-rotation to each grasp
    num_grasps = scaled_grasp_poses.shape[0]
    output_grasps = np.zeros((num_grasps, 7))  # (x, y, z, qw, qx, qy, qz)

    for i in range(num_grasps):
        grasp_quat = scaled_grasp_poses[i, 3:]
        grasp_pos = scaled_grasp_poses[i, :3]

        rot_z = axangles.axangle2mat([0, 0, 1], np.deg2rad(90.0))
        rot_base = quaternions.quat2mat(grasp_quat)  # expects wxyz
        rot = np.matmul(rot_base, rot_z)
        grasp_quat_rotated = quaternions.mat2quat(rot)  # returns wxyz

        output_grasps[i, :3] = grasp_pos
        output_grasps[i, 3:] = grasp_quat_rotated

    return {
        'grasps': output_grasps,
        'closing_angular': valid_closing_angular,
        'closing_linear': valid_closing_linear,
        'shaking_angular': valid_shaking_angular,
        'shaking_linear': valid_shaking_linear,
        'scale_factor': scale_factor,
    }
# 
def compute_gripper_width(
    grasp_pose: np.ndarray,
    points: np.ndarray,
    gripper_z_offset: float = 0.105,
    angle_threshold_deg: float = 15.0,
    distance_threshold: float = 0.05,
    visualize: bool = False,
):
    """Estimate the object width between the fingers along the grasp's y-axis.

    Args:
        grasp_pose: (7,) grasp pose (x, y, z, qw, qx, qy, qz).
        points: (N, 3) object surface points in the same frame.
        gripper_z_offset: Shift along the grasp z-axis to the finger center (m).
        angle_threshold_deg: Unused.
        distance_threshold: Only use points within this radius of the finger center (m).
        visualize: Unused.

    Returns:
        Width in meters, or 0 if no points lie between the fingers.
    """
    # Move grasp forward along local z-axis
    moved_grasp = move_pose_along_local_z(grasp_pose, gripper_z_offset, format='wxyz')

    # Get grasp position and rotation matrix
    grasp_pos = moved_grasp[:3]
    grasp_quat = moved_grasp[3:]  # wxyz
    rot_matrix = quaternions.quat2mat(grasp_quat)  # 3x3 rotation matrix

    # Get Y-axis direction in world frame (column 1 of rotation matrix)
    x_axis = rot_matrix[:, 0]
    y_axis = rot_matrix[:, 1]
    z_axis = rot_matrix[:, 2]

    # Compute vectors from grasp position to all points
    vectors = points - grasp_pos  # (N, 3)
    close_to_gripper_mask = np.linalg.norm(vectors, axis = -1) < distance_threshold
    x_and_z_axis_mask = (np.abs(np.dot(vectors, x_axis)) < 0.005) & (np.abs(np.dot(vectors, z_axis)) < 0.005) 
    mask = close_to_gripper_mask & x_and_z_axis_mask
    vectors = vectors[mask]

    if len(vectors) == 0:
        return 0

    # Compute dot products with Y-axis
    dot_products = np.dot(vectors, y_axis)  # (N,)

    return np.max(dot_products) - np.min(dot_products)

def generate_placement_quaternions(num_quaternions: int = 10, target_xy_angle: float = None):
    """Sample random placement orientations whose z-axis points downward.

    Args:
        num_quaternions: Number of orientations to sample.
        target_xy_angle: If set, keep the z-axis heading in the xy-plane within 45 degrees of this angle (radians).

    Returns:
        (num_quaternions, 4) wxyz quaternions.
    """
    z_world = np.array([0.0, 0.0, 1.0])
    quats = []

    while len(quats) < num_quaternions:
        if target_xy_angle is not None:
            # Constrain z-axis projection onto XY plane to be within 45 degrees of target angle
            angle_offset_rad = np.deg2rad(np.random.uniform(-45, 45))
            xy_angle_rad = target_xy_angle + angle_offset_rad
            
            # Sample z-component (negative to ensure "not-up" hemisphere)
            z_z = np.random.uniform(-1.0, 0.0)  # Negative z component
            
            # Compute x and y components from the angle and z component
            # We want: z_x^2 + z_y^2 + z_z^2 = 1
            # So: z_x^2 + z_y^2 = 1 - z_z^2
            xy_magnitude = np.sqrt(1 - z_z**2)
            z_x = xy_magnitude * np.cos(xy_angle_rad)
            z_y = xy_magnitude * np.sin(xy_angle_rad)
            z = np.array([z_x, z_y, z_z])
        else:
            # Original sampling method
            z = np.random.normal(size=3)
            z /= np.linalg.norm(z)
            
            # Force z into the "not-up" hemisphere:
            if np.dot(z, z_world) > 0.0:
                z = -z

        if np.abs(np.dot(z, z_world)) < 1e-7:
            reference_vec = np.array([1.0, 0.0, 0.0])   # Take x-axis as reference if z is too close to z_world
        else:
            reference_vec = z_world

        y = np.cross(z, reference_vec)
        y /= np.linalg.norm(y)

        # Randomly choose to either flip the y-axis or not
        if np.random.rand() < 0.5:
            y = -y

        x = np.cross(y, z)
        x /= np.linalg.norm(x)

        rot_matrix = np.column_stack([x, y, z])
        
        # Rotate around local z-axis by a random small angle
        angle = np.random.uniform(-1e-3, 1e-3)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_z_local = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        rot_matrix = rot_matrix @ rotation_z_local
        r = R.from_matrix(rot_matrix)
        q = r.as_quat()  # xyzw
        quats.append([q[3], q[0], q[1], q[2]])  # convert to wxyz

    return np.array(quats)


def check_size_validity(object_dir):

    """Check that an ACRONYM object's largest scaled dimension is between 5 and 17 cm.

    Args:
        object_dir: ACRONYM object folder (mesh + annotation.h5).
    """
    mesh_path = os.path.join(object_dir, 'model_repositioned_vhacd.obj')
    grasp_annotation_file = os.path.join(object_dir, 'annotation.h5')

    mesh = trimesh.load(mesh_path)
    
    verts = mesh.vertices
    
    # Find the largest dimension in x and y axes
    x_span = np.max(verts[:, 0]) - np.min(verts[:, 0])
    y_span = np.max(verts[:, 1]) - np.min(verts[:, 1])
    z_span = np.max(verts[:, 2]) - np.min(verts[:, 2])
    
    # Choose the larger dimension
    if x_span >= y_span and x_span >= z_span:
        largest_dimension = x_span
    elif y_span >= x_span and y_span >= z_span:
        largest_dimension = y_span
    else:
        largest_dimension = z_span

    # NOTE: Uncomment this if you want to use the scale factor from the grasp annotation file.
    grasp_annotation = h5py.File(grasp_annotation_file, 'r')
    scale_factor = grasp_annotation['object/scale'][()]

    size_validity = (largest_dimension * scale_factor) > 0.05 and (largest_dimension * scale_factor) < 0.17
    return size_validity

def flip_quat_about_z(quat: np.ndarray):
    """Rotate a wxyz quaternion 180 degrees about its own z-axis.

    Args:
        quat: (4,) wxyz quaternion.
    """
    rot_z = axangles.axangle2mat([0, 0, 1], np.deg2rad(180.0))
    rot_base = quaternions.quat2mat(quat)
    rot = np.matmul(rot_base, rot_z)
    quat_rotated = quaternions.mat2quat(rot)
    return quat_rotated

def flip_pose_quat_about_z(pose: torch.Tensor):
    """Rotate a pose 180 degrees about its own z-axis, keeping its position.

    Args:
        pose: (7,) torch pose (x, y, z, qw, qx, qy, qz).
    """
    # Extract position and quaternion
    position = pose[:3]  # [x, y, z]
    quat_wxyz = pose[3:]  # [qw, qx, qy, qz] in wxyz format
    
    # Convert to numpy for quaternion operations
    quat_np = quat_wxyz.cpu().detach().numpy()
    
    # Flip quaternion around z-axis using the reference function
    quat_flipped_np = flip_quat_about_z(quat_np)
    
    # Convert back to torch tensor
    quat_flipped = torch.tensor(quat_flipped_np, dtype=pose.dtype, device=pose.device)
    
    # Concatenate position and flipped quaternion
    flipped_pose = torch.cat([position, quat_flipped])
    
    return flipped_pose

def build_panda_gripper_grasp_pose_visual(scene: ManiSkillScene):
    """Build a kinematic gripper-shaped marker for showing grasp poses in the scene.

    Args:
        scene: ManiSkill scene to add the marker to.
    """
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual

def point_in_cuboid(corners, point, tol=1e-9):
    """Check whether a point lies inside the axis-aligned box spanned by 8 corners.

    Args:
        corners: (8, 3) cuboid corners, numpy or torch.
        point: (3,) query point, numpy or torch.
        tol: Tolerance added to each bound.
    """
    
    if isinstance(corners, torch.Tensor):
        corners = corners.cpu().numpy()
    if isinstance(point, torch.Tensor):
        point = point.cpu().numpy()
    
    assert len(corners.shape) == 2 and corners.shape[0] == 8 and corners.shape[1] == 3, "Cuboid must have 8 corners and 3 dimensions"
    assert len(point.shape) == 1 and point.shape[0] == 3, "Point must have 3 dimensions"
    
    corners = np.asarray(corners)
    point = np.asarray(point)
    
    # Find axis-aligned bounding box (min/max for each axis)
    x_min = np.min(corners[:, 0])
    x_max = np.max(corners[:, 0])
    y_min = np.min(corners[:, 1])
    y_max = np.max(corners[:, 1])
    z_min = np.min(corners[:, 2])
    z_max = np.max(corners[:, 2])
    
    # Check if point is within bounds (with tolerance)
    return (
        x_min - tol <= point[0] <= x_max + tol and
        y_min - tol <= point[1] <= y_max + tol and
        z_min - tol <= point[2] <= z_max + tol
    )

def to_sapien_pose(pose: Union[np.ndarray, sapien.Pose, torch.Tensor]):
    """Convert a pose to a sapien.Pose.

    Args:
        pose: sapien.Pose, or (7,) array, tensor or list (x, y, z, qw, qx, qy, qz).
    """

    if isinstance(pose, sapien.Pose):
        return pose
    elif isinstance(pose, torch.Tensor):
        return sapien.Pose(p=pose[:3].cpu().numpy(), q=pose[3:].cpu().numpy())
    elif isinstance(pose, np.ndarray):
        return sapien.Pose(p=pose[:3], q=pose[3:])
    elif isinstance(pose, list):
        return sapien.Pose(p=pose[:3], q=pose[3:])
    else:
        raise ValueError(f"Invalid pose type: {type(pose)}")

def to_torch_pose(pose: Union[np.ndarray, sapien.Pose, torch.Tensor], device="cuda:0"):
    """Convert a pose to a (7,) float32 torch tensor.

    Args:
        pose: sapien.Pose, or (7,) array, tensor or list (x, y, z, qw, qx, qy, qz).
        device: Device for the new tensor.
    """
    if isinstance(pose, torch.Tensor):
        return pose
    elif isinstance(pose, np.ndarray):
        return torch.tensor(pose, dtype=torch.float32, device=device)
    elif isinstance(pose, sapien.Pose):
        return torch.tensor(np.concatenate([pose.p, pose.q]), dtype=torch.float32, device=device)
    elif isinstance(pose, list):
        return torch.tensor(np.concatenate([pose[:3], pose[3:]]), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Invalid pose type: {type(pose)}")

def to_numpy_pose(pose: Union[np.ndarray, sapien.Pose, torch.Tensor]):
    """Convert a pose to a (7,) numpy array.

    Args:
        pose: sapien.Pose, or (7,) array, tensor or list (x, y, z, qw, qx, qy, qz).
    """
    if isinstance(pose, torch.Tensor):
        return pose.cpu().numpy()
    elif isinstance(pose, np.ndarray):
        return pose
    elif isinstance(pose, sapien.Pose):
        return np.concatenate([pose.p, pose.q])
    elif isinstance(pose, list):
        return np.concatenate([pose[:3], pose[3:]])
    else:
        raise ValueError(f"Invalid pose type: {type(pose)}")