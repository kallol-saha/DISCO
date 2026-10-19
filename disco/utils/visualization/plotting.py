"""
Open3D and matplotlib views of point clouds, poses and voxel grids.
"""
from typing import Union
# torch is not always required to run stuff from robo_utils
try:
    import torch
except ImportError:
    torch = None
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from .point_cloud_structures import make_gripper_visualization, get_cube_point_cloud
from disco.utils.conversion_utils import pose_to_transformation

# Type hint that works whether torch is available or not
if torch is not None:
    TensorType = Union[np.ndarray, torch.Tensor]
else:
    TensorType = Union[np.ndarray, "torch.Tensor"]  # String literal for when torch is not available

COLORS = {
    "blue": np.array([78, 121, 167]) / 255.0,  # blue
    "green": np.array([89, 161, 79]) / 255.0,  # green
    "brown": np.array([156, 117, 95]) / 255.0,  # brown
    "orange": np.array([242, 142, 43]) / 255.0,  # orange
    "yellow": np.array([237, 201, 72]) / 255.0,  # yellow
    "gray": np.array([186, 176, 172]) / 255.0,  # gray
    "red": np.array([255, 87, 89]) / 255.0,  # red
    "purple": np.array([176, 122, 161]) / 255.0,  # purple
    "cyan": np.array([118, 183, 178]) / 255.0,  # cyan
    "pink": np.array([255, 157, 167]) / 255.0,
    "prediction": np.array([153, 255, 51]) / 255.0,
    "action": np.array([0, 128, 255]) / 255.0,
}

def reshape_to_points(data):
    """Return a writable (-1, 3) copy of an array, moving its size-3 axis last.

    Args:
        data: 2D or 3D numpy array with one axis of length 3.
    """
    # Ensure data has 2-3 dimensions
    if len(data.shape) > 3 or len(data.shape) < 2:
        raise ValueError("Data must have 2 or 3 dimensions")

    # Find dimension with length 3 and move it to end
    three_dim = None
    for i, dim in enumerate(data.shape):
        if dim == 3:
            three_dim = i
            break
    
    if three_dim is None:
        raise ValueError("Data must have one dimension of length 3")
        
    # Move dimension with length 3 to end and reshape
    if three_dim != len(data.shape)-1:
        dims = list(range(len(data.shape)))
        dims.remove(three_dim)
        dims.append(three_dim)
        data = np.transpose(data, dims)
    
    data = data.reshape(-1, 3)
    
    # Create writable copy
    data_new = np.zeros(data.shape)
    data_new[:] = data[:]
    
    return data_new

def plot_pcd(pcd, colors=None, seg=None, base_frame=False, extra_frames=None, frame_size=0.2):
    """Show a point cloud in an Open3D window.

    Args:
        pcd: (N, 3) points, numpy or torch.
        colors: (N, 3) RGB colors in [0, 1] or [0, 255]; black if None.
        seg: (N,) or (N, 1) segment ids, used for coloring when colors is None.
        base_frame: Also draw the world coordinate frame.
        extra_frames: List of (4, 4) transforms to draw as coordinate frames.
        frame_size: Axis length of the drawn frames.
    """

    if torch is not None and type(pcd) == torch.Tensor:
        pcd = pcd.cpu().detach().numpy()
    if colors is not None and torch is not None and type(colors) == torch.Tensor:
        colors = colors.cpu().detach().numpy()
    if seg is not None:
        if torch is not None and type(seg) == torch.Tensor:
            seg = seg.cpu().detach().numpy()
        seg = seg.flatten()

    # Reshape point cloud to (-1, 3) and create writable copy
    pcd_new = reshape_to_points(pcd)

    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd_new)

    if colors is not None:
        # Apply the same reshaping to colors as we did to pcd
        colors_new = reshape_to_points(colors)
        # Ensure colors are in the right range [0, 1]
        if colors_new.max() > 1.0:
            colors_new = colors_new / 255.0
        pts_vis.colors = o3d.utility.Vector3dVector(colors_new)

    elif seg is not None:
        seg_ids = np.unique(seg)
        n = len(seg_ids)
        cmap = plt.get_cmap("tab10")
        id_to_color = {uid: cmap(i / n)[:3] for i, uid in enumerate(seg_ids)}
        colors = np.array([id_to_color[uid] for uid in seg])
        pts_vis.colors = o3d.utility.Vector3dVector(colors)

    else:
        # Set default black color if no colors provided
        black_colors = np.zeros((pcd_new.shape[0], 3))
        pts_vis.colors = o3d.utility.Vector3dVector(black_colors)

    geometries = [pts_vis]

    if base_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size, origin=[0, 0, 0]
        )
        geometries.append(frame)

    if extra_frames is not None:
        for frame_transform in extra_frames:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frame_size, origin=[0, 0, 0]
            )
            frame.transform(frame_transform)
            geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)

def visualize_poses_in_pointcloud(pcd, poses, rgb=None, colors=None, frame_size=0.2, length=0.05, density=500, extra_frames=None):
    """Show gripper markers at the given poses inside a point cloud.

    Args:
        pcd: (N, 3) points.
        poses: List of (7,) wxyz poses, numpy or torch.
        rgb: (N, 3) point colors; black if None.
        colors: Per-pose RGB colors, cycled; red if None.
        frame_size: Axis length of the drawn frames.
        length: Half-width of each gripper marker.
        density: Points per marker line.
        extra_frames: List of (4, 4) transforms to draw as coordinate frames.
    """
    if colors is None:
        colors = [(1, 0, 0)] * len(poses)

    combined_pcd = pcd.copy()
    combined_rgb = rgb.copy() if rgb is not None else np.zeros_like(pcd)

    for i, pose in enumerate(poses):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        gripper_transform = pose_to_transformation(pose, format='wxyz')
        gripper_points, gripper_colors = make_gripper_visualization(
            rotation=gripper_transform[:3, :3],
            translation=gripper_transform[:3, 3],
            length=length,
            density=density,
            color=colors[i % len(colors)]
        )
        combined_pcd = np.vstack([combined_pcd, gripper_points])
        combined_rgb = np.vstack([combined_rgb, gripper_colors])

    if combined_rgb is not None:
        plot_pcd(combined_pcd, combined_rgb, base_frame=True, frame_size=frame_size, extra_frames=extra_frames)
    else:
        plot_pcd(combined_pcd, base_frame=True, frame_size=frame_size, extra_frames=extra_frames)

def plot_pcd_with_highlighted_segment(pcd, seg, segment_id):
    """Show a point cloud with one segment highlighted.

    Args:
        pcd: (N, 3) points.
        seg: (N,) or (N, 1) segment ids.
        segment_id: Segment to highlight.
    """
    if seg is not None:
        if torch is not None and type(seg) == torch.Tensor:
            seg = seg.cpu().detach().numpy()
        seg = seg.flatten()

    binary_seg = np.where(seg == segment_id, 1, 0)
    if torch is None:
        raise ImportError("torch is not available. This function requires torch.")
    seg = torch.from_numpy(binary_seg).unsqueeze(-1)

    plot_pcd(pcd, seg=seg)

def plot_voxel_grid_with_action(voxel_grid: TensorType, 
                    action_voxels: TensorType,
                    action_colors: TensorType):
    """Show the occupied voxels of a grid with the action voxels in their own colors.

    Args:
        voxel_grid: (10, D, H, W) torch voxel features (3:6 RGB, 6:9 voxel center).
        action_voxels: (N, 3) voxel indices of the actions.
        action_colors: (N, 3) RGB colors of the action voxels.
    """
    if torch is None:
        raise ImportError("torch is not available. This function requires torch.")
    
    vis_grid = voxel_grid.permute(1, 2, 3, 0)

    # Remove the action voxels from the voxel grid:
    vis_grid[action_voxels[:, 0], 
             action_voxels[:, 1], 
             action_voxels[:, 2], 
             :3] = 0.

    # Mask out the points that are not in the voxel grid:
    mask = torch.norm(vis_grid[..., :3], dim = -1) > 0      # Could just use occupancy instead...
    vis_pts = vis_grid[torch.where(mask)][..., 6:9]
    vis_rgb = vis_grid[torch.where(mask)][..., 3:6]

    # Add the action voxels to the voxel grid point cloud
    action_voxel_center = vis_grid[action_voxels[:, 0], 
                                   action_voxels[:, 1], 
                                   action_voxels[:, 2], 
                                   6:9]
    
    vis_pts = torch.cat([vis_pts, action_voxel_center], dim=0)
    vis_rgb = torch.cat([vis_rgb, action_colors], dim=0)

    plot_pcd(vis_pts, vis_rgb)

def create_cube_without_points(cube_size=64):
    """Return an Open3D line set of grey cube edges spanning [0, cube_size]^3.

    Args:
        cube_size: Cube edge length.
    """
    # Create the cube edges (12 edges of a cube)
    cube_points = [
        [0, 0, 0],           # 0: bottom front left
        [cube_size, 0, 0],   # 1: bottom front right
        [cube_size, cube_size, 0], # 2: bottom back right
        [0, cube_size, 0],   # 3: bottom back left
        [0, 0, cube_size],   # 4: top front left
        [cube_size, 0, cube_size], # 5: top front right
        [cube_size, cube_size, cube_size], # 6: top back right
        [0, cube_size, cube_size]  # 7: top back left
    ]
    
    # Define the 12 edges of the cube
    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    # Create line set for cube edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(cube_points)
    line_set.lines = o3d.utility.Vector2iVector(cube_edges)
    line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(cube_edges))  # Grey edges
    
    return line_set

def plot_prediction(data_path, scene_bounds=None, gripper_length=0.05, gripper_density=300, 
                   gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """Show a saved scene with its ground-truth and predicted gripper poses.

    Args:
        data_path: .npz file with rgb, pcd, pred and gt arrays (poses use xyzw quats).
        scene_bounds: Crop box [x_min, y_min, z_min, x_max, y_max, z_max], or None.
        gripper_length: Half-width of each gripper marker.
        gripper_density: Points per marker line.
        gt_color: RGB color of the ground-truth gripper.
        pred_color: RGB color of the predicted gripper.
    """
    data = np.load(data_path)
    rgb = data["rgb"]
    pcd = data["pcd"]
    pred = data["pred"]
    gt = data["gt"]

    # ------------ Create the Scene Point Cloud ------------ #

    # Combine the different views along the first dimension
    rgb = np.transpose(rgb, (0, 2, 3, 1))  # (num_views, 3, H, W) -> (num_views, H, W, 3)
    pcd = np.transpose(pcd, (0, 2, 3, 1))  # (num_views, 3, H, W, 3) -> (num_views, H, W, 3)
    rgb = rgb.reshape(-1, 3)  # (num_views, H, W, 3) -> (num_views*H*W, 3)
    pcd = pcd.reshape(-1, 3)  # (num_views, H, W, 3) -> (num_views*H*W, 3)

    # Apply scene bounds filtering if provided
    if scene_bounds is not None:
        bound_mask = (pcd[:, 0] > scene_bounds[0]) & (pcd[:, 0] < scene_bounds[3]) & \
                     (pcd[:, 1] > scene_bounds[1]) & (pcd[:, 1] < scene_bounds[4]) & \
                     (pcd[:, 2] > scene_bounds[2]) & (pcd[:, 2] < scene_bounds[5])
        rgb = rgb[bound_mask]
        pcd = pcd[bound_mask]

    # -------------------------------------------------------- #

    # ------------ Create the Gripper Point Clouds ------------ #

    # GROUND TRUTH:
    gt_pos = gt[:3]
    gt_quat = gt[3:7]
    gt_rot = R.from_quat(gt_quat).as_matrix()

    # PREDICTION:
    pred_pos = pred[:3]
    pred_quat = pred[3:7]
    pred_rot = R.from_quat(pred_quat).as_matrix()

    gt_gripper_points, gt_gripper_colors = make_gripper_visualization(
        rotation=gt_rot,
        translation=gt_pos,
        length=gripper_length,
        density=gripper_density,
        color=gt_color
    )

    pred_gripper_points, pred_gripper_colors = make_gripper_visualization(
        rotation=pred_rot,
        translation=pred_pos,
        length=gripper_length,
        density=gripper_density,
        color=pred_color
    )

    # --------------------------------------------------------- #

    # Combine scene points with gripper points
    pcd = np.vstack([pcd, gt_gripper_points, pred_gripper_points])
    rgb = np.vstack([rgb, gt_gripper_colors, pred_gripper_colors])

    # Plot the point clouds
    plot_pcd(pcd, colors=rgb)

def voxel_points_and_features_from_voxel_grid(voxel_grid, action_voxels, action_colors):
    """Convert a voxel grid to points and colors, drawing action voxels as colored cubes.

    Args:
        voxel_grid: (F, X, Y, Z) or (X, Y, Z, F) features (3:6 RGB in [-1, 1], 6:9 center, last occupancy).
        action_voxels: (K, 3) voxel indices of the actions.
        action_colors: (K, 3) RGB colors of the action voxels.

    Returns:
        (N, 3) points and (N, 3) RGB colors in [0, 1].
    """

    # Convert inputs to numpy arrays if they are torch tensors
    if torch is not None and torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.detach().cpu().numpy()
    if torch is not None and torch.is_tensor(action_voxels):
        action_voxels = action_voxels.detach().cpu().numpy()
    if torch is not None and torch.is_tensor(action_colors):
        action_colors = action_colors.detach().cpu().numpy()
        
    # Convert voxel grid to (X, Y, Z, features) format
    if voxel_grid.shape[0] < voxel_grid.shape[-1]:  # If in (features, X, Y, Z) format
        vis_grid = np.transpose(voxel_grid, (1, 2, 3, 0))
    else:
        vis_grid = voxel_grid
    
    # Remove the action voxels from the voxel grid for point cloud visualization
    for i in range(action_voxels.shape[0]):
        vis_grid[action_voxels[i, 0], 
                 action_voxels[i, 1], 
                 action_voxels[i, 2], 
                 -1] = 0.
    
    # Mask out the points that are not in the voxel grid
    mask = (vis_grid[..., -1] == 1)       #np.linalg.norm(vis_grid[..., :3], axis=-1) > 0
    points = vis_grid[np.where(mask)][..., 6:9]  # Voxel center coordinates
    features = vis_grid[np.where(mask)][..., 3:6]  # RGB colors
    
    # Go from -1 to 1 range to 0 to 1 range (for RGB)
    features = (features + 1) / 2

    for i in range(action_voxels.shape[0]):
        # Get action voxel center coordinates
        action_voxel_center = vis_grid[action_voxels[i, 0], 
                                        action_voxels[i, 1], 
                                        action_voxels[i, 2], 
                                        6:9]  # Voxel center coordinates
        
        # Compute how the voxel center z-coordinate (-2 element) is shifted when moving along the z-indices of the grid
        voxel_size = (voxel_grid[-2, 0, 0, 1] - voxel_grid[-2, 0, 0, 0]) * 2
        action_points, action_features = get_cube_point_cloud(voxel_size, action_voxel_center, action_colors[i])

        points = np.vstack((points, action_points))
        features = np.vstack((features, action_features))

    return points, features

def visualize_image(pcd_vis, point_size=0.4, zoom_out=0.1):
    """Render an Open3D geometry offscreen to a 512x512 image.

    Args:
        pcd_vis: Open3D geometry to render.
        point_size: Rendered point size.
        zoom_out: Camera offset along z.

    Returns:
        (512, 512, 3) uint8 RGB image.
    """
    # Set up Open3D visualization and render
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=512, height=512)
    vis.add_geometry(pcd_vis)

    # Set point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.eye(4)
    extrinsic[2, 3] = zoom_out  # Zoom out to capture everything
    camera_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Render and capture the view
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)

    # Convert image to numpy array
    image_np = (np.asarray(image) * 255).astype(np.uint8)

    # Clean up
    vis.destroy_window()
    return image_np