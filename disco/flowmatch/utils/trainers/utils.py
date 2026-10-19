import torch
import open3d as o3d
import numpy as np


def compute_metrics(pred, gt):
    # pred/gt are (B, L, 3+rot+1)
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    # symmetric quaternion eval
    quat_l1 = (pred[..., 3:-1] - gt[..., 3:-1]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:-1] + gt[..., 3:-1]).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
    # gripper openess
    openess = ((pred[..., -1:] >= 0.5) == (gt[..., -1:] >= 0.5)).bool()
    tr = 'traj_'

    # Trajectory metrics
    ret_1, ret_2 = {
        tr + 'pos_l2': pos_l2.mean(),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
        tr + 'rot_l1': quat_l1.mean(),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
        tr + 'gripper': openess.flatten().float().mean()
    }, {
        tr + 'pos_l2': pos_l2.mean(-1),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
        tr + 'rot_l1': quat_l1.mean(-1),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
    }

    return ret_1, ret_2

def visualize_pred_gt(rgb, pcd, pred, gt, step_id, log_dir):
    """
    Save visualization data as numpy arrays in NPZ files.
    
    Args:
        rgb: (num_views, 3, H, W)
        pcd: (num_views, 3, H, W)
        pred: (B, L, 3+rot+1)
        gt: (B, L, 3+rot+1)
        step_id: Current step ID for filename
        log_dir: Directory to save the NPZ files
    """
    import os
    
    # Create vis directory if it doesn't exist
    vis_dir = os.path.join(log_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Convert to numpy and detach from GPU
    if type(rgb) == torch.Tensor:
        rgb_np = rgb.cpu().detach().numpy()
    else:
        rgb_np = rgb
        
    if type(pcd) == torch.Tensor:
        pcd_np = pcd.cpu().detach().numpy()
    else:
        pcd_np = pcd
        
    if type(pred) == torch.Tensor:
        pred_np = pred.cpu().detach().numpy()
    else:
        pred_np = pred
        
    if type(gt) == torch.Tensor:
        gt_np = gt.cpu().detach().numpy()
    else:
        gt_np = gt
    
    # Save as NPZ file
    filename = os.path.join(vis_dir, f"step_{step_id:06d}.npz")
    np.savez_compressed(
        filename,
        rgb=rgb_np,
        pcd=pcd_np,
        pred=pred_np,
        gt=gt_np,
        step_id=step_id
    )
    
    print(f"Saved visualization data to: {filename}")
    print(f"  RGB shape: {rgb_np.shape}")
    print(f"  PCD shape: {pcd_np.shape}")
    print(f"  Pred shape: {pred_np.shape}")
    print(f"  GT shape: {gt_np.shape}")

def reshape_to_points(data):
    """
    Reshape data to have shape (-1, 3) by finding the dimension with length 3
    and moving it to the end, then flattening all other dimensions.
    Also creates a writable copy of the data.
    
    Args:
        data: numpy array with one dimension of length 3
        
    Returns:
        writable reshaped data with shape (-1, 3)
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

def plot_pcd(pcd, colors=None, frame=False):

    if type(pcd) == torch.Tensor:
        pcd = pcd.cpu().detach().numpy()
    if colors is not None and type(colors) == torch.Tensor:
        colors = colors.cpu().detach().numpy()

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

    geometries = [pts_vis]

    if frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)


def plot_pcd_with_frame(pcd, colors=None, poses=None):
    """
    Plot point cloud with home frame and multiple custom frames specified by poses.
    
    Args:
        pcd: point cloud data (torch.Tensor or numpy array)
        colors: point cloud colors (optional)
        poses: list of pose tensors, each of shape [1, 1, 8] or [1, 8] containing [x, y, z, qx, qy, qz, qw, gripper]
    """
    if type(pcd) == torch.Tensor:
        pcd = pcd.cpu().detach().numpy()
    if colors is not None and type(colors) == torch.Tensor:
        colors = colors.cpu().detach().numpy()

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

    geometries = [pts_vis]

    # Always add home frame
    home_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )
    geometries.append(home_frame)

    # Add custom frames if poses are provided
    if poses is not None:
        if not isinstance(poses, list):
            poses = [poses]  # Convert single pose to list
        
        for pose in poses:
            if type(pose) == torch.Tensor:
                pose = pose.cpu().detach().numpy()
            
            # Handle different pose shapes
            if pose.shape == (1, 1, 8):
                pose = pose[0, 0, :]  # Extract [8]
            elif pose.shape == (1, 8):
                pose = pose[0, :]  # Extract [8]
            elif pose.shape == (8,):
                pass  # Already correct shape
            else:
                raise ValueError(f"Pose must have shape [1, 1, 8], [1, 8], or [8], got {pose.shape}")
            
            # Extract position and quaternion
            position = pose[:3]  # [x, y, z]
            quaternion = pose[3:7]  # [qx, qy, qz, qw]
            
            # Create rotation matrix from quaternion
            # Open3D expects quaternion in [w, x, y, z] format
            quat_o3d = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # [w, x, y, z]
            
            # Create custom frame
            custom_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.15, origin=position
            )
            
            # Apply rotation
            R = custom_frame.get_rotation_matrix_from_quaternion(quat_o3d)
            custom_frame.rotate(R, center=position)
            
            geometries.append(custom_frame)

    o3d.visualization.draw_geometries(geometries)


def plot_pcd_rgb_rotating_matplotlib(pcd, rgb=None, poses=None, output_file="rotating_pcd.gif", 
                                   fps=30, duration=3, point_size=1, subsample_factor=1):
    """
    Create a rotating GIF visualization of point cloud with RGB colors using matplotlib.
    
    Args:
        pcd: point cloud data (torch.Tensor or numpy array) of shape (N, 3) or (H, W, 3)
        rgb: RGB color data (torch.Tensor or numpy array) of same shape as pcd
        poses: list of pose tensors for coordinate frames (optional)
        output_file: Output GIF file path
        fps: Frames per second
        duration: Duration of GIF in seconds
        point_size: Size of points in scatter plot
        subsample_factor: Factor to subsample points (1 = all points, 2 = every 2nd point, etc.)
    """
    import matplotlib
    matplotlib.use('Agg')  # Set backend for headless operation
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # Convert to numpy if needed
    if type(pcd) == torch.Tensor:
        pcd = pcd.cpu().detach().numpy()
    if rgb is not None and type(rgb) == torch.Tensor:
        rgb = rgb.cpu().detach().numpy()

    # Reshape point cloud to (-1, 3)
    pcd_reshaped = reshape_to_points(pcd)
    
    # Subsample points if requested
    if subsample_factor > 1:
        indices = np.arange(0, len(pcd_reshaped), subsample_factor)
        pcd_reshaped = pcd_reshaped[indices]
        if rgb is not None:
            rgb_reshaped = reshape_to_points(rgb)[indices]
        else:
            rgb_reshaped = None
    else:
        if rgb is not None:
            rgb_reshaped = reshape_to_points(rgb)
        else:
            rgb_reshaped = None

    # Calculate number of frames
    total_frames = fps * duration
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate axis limits from point cloud
    pcd_min = pcd_reshaped.min(axis=0)
    pcd_max = pcd_reshaped.max(axis=0)
    pcd_range = pcd_max - pcd_min
    center = (pcd_min + pcd_max) / 2
    
    # Set axis limits with some padding
    padding = pcd_range.max() * 0.1
    ax.set_xlim([pcd_min[0] - padding, pcd_max[0] + padding])
    ax.set_ylim([pcd_min[1] - padding, pcd_max[1] + padding])
    ax.set_zlim([pcd_min[2] - padding, pcd_max[2] + padding])
    
    # Remove grid and axes walls for clean background
    ax.grid(False)
    ax.set_axis_off()
    
    # Set background to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Set initial view angle
    ax.view_init(elev=20, azim=0)
    
    # Create scatter plot of point cloud
    if rgb_reshaped is not None:
        # Ensure colors are in the right range [0, 1]
        if rgb_reshaped.max() > 1.0:
            rgb_reshaped = rgb_reshaped / 255.0
        scatter = ax.scatter(pcd_reshaped[:, 0], pcd_reshaped[:, 1], pcd_reshaped[:, 2], 
                           c=rgb_reshaped, s=point_size, alpha=0.8)
    else:
        scatter = ax.scatter(pcd_reshaped[:, 0], pcd_reshaped[:, 1], pcd_reshaped[:, 2], 
                           c='blue', s=point_size, alpha=0.8)
    
    # Add coordinate frames if poses are provided
    quiver_objects = []
    if poses is not None:
        if not isinstance(poses, list):
            poses = [poses]
        
        for i, pose in enumerate(poses):
            if type(pose) == torch.Tensor:
                pose = pose.cpu().detach().numpy()
            
            # Handle different pose shapes
            if pose.shape == (1, 1, 8):
                pose = pose[0, 0, :]
            elif pose.shape == (1, 8):
                pose = pose[0, :]
            elif pose.shape == (8,):
                pass
            else:
                raise ValueError(f"Pose must have shape [1, 1, 8], [1, 8], or [8], got {pose.shape}")
            
            # Extract position and quaternion
            position = pose[:3]
            quaternion = pose[3:7]
            
            # Create coordinate frame arrows
            frame_size = pcd_range.max() * 0.1  # Scale frame to point cloud size
            origin = position
            
            # X-axis (red)
            quiver_x = ax.quiver(origin[0], origin[1], origin[2], 
                               frame_size, 0, 0, 
                               color='red', arrow_length_ratio=0.2, linewidth=2)
            # Y-axis (green)
            quiver_y = ax.quiver(origin[0], origin[1], origin[2], 
                               0, frame_size, 0, 
                               color='green', arrow_length_ratio=0.2, linewidth=2)
            # Z-axis (blue)
            quiver_z = ax.quiver(origin[0], origin[1], origin[2], 
                               0, 0, frame_size, 
                               color='blue', arrow_length_ratio=0.2, linewidth=2)
            
            quiver_objects.extend([quiver_x, quiver_y, quiver_z])
    
    # Function to update the plot for each frame
    def update(frame):
        ax.view_init(elev=20, azim=frame * 360 / total_frames)
        return [scatter] + quiver_objects
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=True)
    
    # Save as GIF
    anim.save(output_file, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Created rotating point cloud GIF: {output_file}")
    print(f"Point cloud shape: {pcd_reshaped.shape}")
    if rgb_reshaped is not None:
        print(f"RGB shape: {rgb_reshaped.shape}")
    print(f"Subsampled by factor: {subsample_factor}")

    
