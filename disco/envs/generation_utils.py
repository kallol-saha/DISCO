"""
Helpers for generating shelf environments: shelf meshes and object placement sampling.
"""
import os
import xml.etree.ElementTree as ET
import trimesh
import numpy as np
import pybullet as p
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import matplotlib.pyplot as plt

GRID_SIZE = 0.001

p.connect(p.DIRECT)

# ------------------ MESH PROCESSING UTILS ------------------ #

def process_mesh_with_vhacd():
    """Run V-HACD convex decomposition on the raw shelf back-wall mesh (hardcoded paths)."""
    # Load the original mesh
    input_file = "assets/shelf/back_wall_raw.obj"
    output_file = "assets/shelf/back_wall.obj"
    log_file = "vhacd_log.txt"    
    
    p.vhacd(input_file, output_file, log_file)
    
    print(f"Original mesh processed with vhacd")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")

def scale_urdf_meshes(urdf_file, scale_factor, output_file):
    """Multiply every mesh scale in a URDF and save the result.

    Args:
        urdf_file: Input URDF path.
        scale_factor: Factor applied to each mesh scale (missing scales count as 1).
        output_file: Output URDF path.
    """
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        scale = mesh.get("scale")
        if scale:
            current_scale = [float(x) for x in scale.split()]
            new_scale = [x * scale_factor for x in current_scale]
        else:
            # Default scale is 1,1,1 if not defined
            new_scale = [scale_factor] * 3

        mesh.set("scale", " ".join(f"{x:.6f}" for x in new_scale))

    tree.write(output_file, encoding="utf-8", xml_declaration=True)


# ------------------ SHELF GENERATION UTILS ------------------ #

def save_shelf_urdf(
    shelf_height,
    shelf_width, 
    shelf_depth, 
    wall_thickness, 
    shelf_distance_from_floor,
    bottom_wall_height,
    shelf_folder = "assets/shelf", 
    color = (0.6, 0.4, 0.2), 
    urdf_filename="shelf.urdf"
):
    """Write a URDF that joins the five shelf wall meshes with fixed joints.

    Args:
        shelf_height: Inner height of the shelf, in meters.
        shelf_width: Inner width (y) of the shelf, in meters.
        shelf_depth: Depth (x) of the shelf, in meters.
        wall_thickness: Wall thickness, in meters.
        shelf_distance_from_floor: Height of the shelf floor above the ground, in meters.
        bottom_wall_height: Thickness of the bottom block, in meters.
        shelf_folder: Folder holding the wall meshes; the URDF is saved here too.
        color: RGB color of the shelf, each in [0, 1].
        urdf_filename: Output URDF file name.
    """
    urdf_file_path = os.path.join(shelf_folder, urdf_filename)

    """
    Generates a URDF file for a shelf with multiple walls connected by fixed joints.

    :param shelf_folder: Folder containing the shelf mesh files.
    :param color: A tuple of length 3 representing the RGB color of the shelf.
    :param urdf_filename: Name of the output URDF file.
    """

    # shelf_position = [
    #     0., 
    #     0., 
    #     shelf_distance_from_floor
    #     ]

    bottom_wall_position = [
        0., 
        0., 
        shelf_distance_from_floor - bottom_wall_height
        ]

    left_wall_position = [
        0., 
        -(shelf_width / 2 + wall_thickness / 2), 
        0.
        ]
    right_wall_position = [
        0., 
        shelf_width / 2 + wall_thickness / 2, 
        0.
        ]
    back_wall_position = [
        (shelf_depth / 2 + wall_thickness / 2), 
        0., 
        0.
        ]
    top_wall_position = [
        0., 
        0., 
        shelf_distance_from_floor + shelf_height + wall_thickness
        ]

    urdf_content = f'''<?xml version="1.0"?>
<robot name="dummy">
  <link name="bottom_wall">
    <visual>
      <origin xyz="{bottom_wall_position[0]} {bottom_wall_position[1]} {bottom_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="bottom_wall.obj" scale="1 1 1"/>
      </geometry>
      <material name="brown">
        <color rgba="{color[0]} {color[1]} {color[2]} 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{bottom_wall_position[0]} {bottom_wall_position[1]} {bottom_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="bottom_wall.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <link name="left_wall">
    <visual>
      <origin xyz="{left_wall_position[0]} {left_wall_position[1]} {left_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="vertical_wall.obj" scale="1 1 1"/>
      </geometry>
      <material name="brown">
        <color rgba="{color[0]} {color[1]} {color[2]} 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{left_wall_position[0]} {left_wall_position[1]} {left_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="vertical_wall.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <link name="right_wall">
    <visual>
      <origin xyz="{right_wall_position[0]} {right_wall_position[1]} {right_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="vertical_wall.obj" scale="1 1 1"/>
      </geometry>
      <material name="brown">
        <color rgba="{color[0]} {color[1]} {color[2]} 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{right_wall_position[0]} {right_wall_position[1]} {right_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="vertical_wall.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <link name="back_wall">
    <visual>
      <origin xyz="{back_wall_position[0]} {back_wall_position[1]} {back_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="back_wall.obj" scale="1 1 1"/>
      </geometry>
      <material name="brown">
        <color rgba="{color[0]} {color[1]} {color[2]} 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{back_wall_position[0]} {back_wall_position[1]} {back_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="back_wall.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <link name="top_wall">
    <visual>
      <origin xyz="{top_wall_position[0]} {top_wall_position[1]} {top_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="top_wall.obj" scale="1 1 1"/>
      </geometry>
      <material name="brown">
        <color rgba="{color[0]} {color[1]} {color[2]} 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{top_wall_position[0]} {top_wall_position[1]} {top_wall_position[2]}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="top_wall.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wall_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="bottom_wall"/>
    <child link="left_wall"/>
  </joint>

  <joint name="right_wall_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="bottom_wall"/>
    <child link="right_wall"/>
  </joint>

  <joint name="back_wall_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="bottom_wall"/>
    <child link="back_wall"/>
  </joint>

  <joint name="top_wall_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="bottom_wall"/>
    <child link="top_wall"/>
  </joint>

</robot>'''

    # Save to file
    with open(urdf_file_path, "w") as f:
        f.write(urdf_content)

    print(f"URDF saved as {urdf_file_path}")

def create_shelf(
    mesh_folder = "assets/shelf",
    shelf_width = 0.6, 
    shelf_height = 0.8, 
    shelf_depth = 0.3, 
    wall_thickness = 0.01, 
    shelf_distance_from_floor = 0.4,
    bottom_wall_height = 0.088,
    color = (0.6, 0.4, 0.2) # Brown
):   
    """Build, convex-decompose and save the shelf wall meshes, then write the shelf URDF.

    Args:
        mesh_folder: Output folder for the meshes and URDF.
        shelf_width: Inner width (y) of the shelf, in meters.
        shelf_height: Inner height of the shelf, in meters.
        shelf_depth: Depth (x) of the shelf, in meters.
        wall_thickness: Wall thickness, in meters.
        shelf_distance_from_floor: Height of the shelf floor above the ground, in meters.
        bottom_wall_height: Thickness of the bottom block, in meters.
        color: RGB color of the shelf, each in [0, 1].
    """

    horizontal_rectangle = Polygon([(-shelf_depth / 2, -shelf_width / 2), (-shelf_depth / 2, shelf_width / 2), 
                    (shelf_depth / 2, shelf_width / 2), (shelf_depth / 2, -shelf_width / 2)])

    vertical_rectangle = Polygon([(-shelf_depth / 2, -wall_thickness / 2), (-shelf_depth / 2, wall_thickness / 2), 
                    (shelf_depth / 2, wall_thickness / 2), (shelf_depth / 2, -wall_thickness / 2)])

    back_rectangle = Polygon([(-wall_thickness / 2, -shelf_width / 2), (-wall_thickness / 2, shelf_width / 2), 
                    (wall_thickness / 2, shelf_width / 2), (wall_thickness / 2, -shelf_width / 2)])

    full_height = shelf_distance_from_floor + (1.5 * wall_thickness) + shelf_height
    
    top_wall_mesh = trimesh.creation.extrude_polygon(horizontal_rectangle, height = wall_thickness)
    vertical_wall_mesh = trimesh.creation.extrude_polygon(vertical_rectangle, height = full_height)
    back_wall_mesh = trimesh.creation.extrude_polygon(back_rectangle, height = full_height)
    bottom_wall_mesh = trimesh.creation.extrude_polygon(horizontal_rectangle, height = bottom_wall_height)

    # Make the directory where the meshes will be saved
    os.makedirs(mesh_folder, exist_ok=True)
    
    # Save the meshes into OBJ files
    top_wall_mesh.export(os.path.join(mesh_folder, "top_wall.obj"))
    vertical_wall_mesh.export(os.path.join(mesh_folder, "vertical_wall.obj"))
    back_wall_mesh.export(os.path.join(mesh_folder, "back_wall.obj"))
    bottom_wall_mesh.export(os.path.join(mesh_folder, "bottom_wall.obj"))

    # Run V-HACD on the meshes and save them again
    p.vhacd(os.path.join(mesh_folder, "top_wall.obj"), os.path.join(mesh_folder, "top_wall.obj"), "log.txt")
    p.vhacd(os.path.join(mesh_folder, "vertical_wall.obj"), os.path.join(mesh_folder, "vertical_wall.obj"), "log.txt")
    p.vhacd(os.path.join(mesh_folder, "back_wall.obj"), os.path.join(mesh_folder, "back_wall.obj"), "log.txt")
    p.vhacd(os.path.join(mesh_folder, "bottom_wall.obj"), os.path.join(mesh_folder, "bottom_wall.obj"), "log.txt")

    # Save the URDF file
    save_shelf_urdf(
        shelf_height, 
        shelf_width, 
        shelf_depth, 
        wall_thickness, 
        shelf_distance_from_floor, 
        bottom_wall_height,
        mesh_folder, 
        color,
        urdf_filename="shelf.urdf")


# ------------------ CAMERA POSITION SAMPLING UTILS ------------------ #

def compute_ray_box_intersection(
  ray_origin,         # (x, y, z) origin, must be inside box bounds
  azimuth_angle,      # Orientation angle in XY plane in radians
  elevation_angle,    # Elevation angle in radians
  box_min,            # (x_min, y_min, z_min) bounds
  box_max             # (x_max, y_max, z_max) bounds
):
  """Return where a ray starting inside a box leaves it.

  Args:
    ray_origin: (x, y, z) ray start; must be inside the box.
    azimuth_angle: Ray heading in the XY plane, in radians.
    elevation_angle: Ray elevation above the XY plane, in radians.
    box_min: (x_min, y_min, z_min) box corner.
    box_max: (x_max, y_max, z_max) box corner.

  Returns:
    (3,) exit point, or None if the ray does not exit going forward.
  """

  # Assert that ray_origin is inside box bounds
  assert ray_origin[0] >= box_min[0] and ray_origin[0] <= box_max[0]
  assert ray_origin[1] >= box_min[1] and ray_origin[1] <= box_max[1]
  assert ray_origin[2] >= box_min[2] and ray_origin[2] <= box_max[2]

  # Random point inside cuboid
  p0 = np.array(ray_origin)
  box_min = np.array(box_min)
  box_max = np.array(box_max)

  # Direction vector of ray
  d = np.array([
      np.cos(elevation_angle) * np.cos(azimuth_angle),
      np.cos(elevation_angle) * np.sin(azimuth_angle),
      np.sin(elevation_angle)
  ])

  # 1. Subtacting the max and min corners from the point gets us the vectors towards corners with the point as origin
  # 2. Element-wise dividing these vectors by the direction vector gives us the number of steps the 
  # ray takes along each axis to reach each of the bounds corresponding to the axes.
  tmin = (box_min - p0) / d
  tmax = (box_max - p0) / d

  # Element-wise minimum gives us the number of backward steps, so that is when the ray enters the box.
  # Element-wise maximum gives us the number of forward steps, so that is when the ray exits the box.
  t1 = np.minimum(tmin, tmax)
  t2 = np.maximum(tmin, tmax)

  # Maximum of the backward steps gives us the number of steps the ray takes to enter the box.
  # Minimum of the forward steps gives us the number of steps the ray takes to exit the box.
  t_enter = np.max(t1)
  t_exit = np.min(t2)

  if t_exit < 0 or t_enter > t_exit:
      # The ray is either going backward or the exit point is before the enter point.
      return None
  else:
      # The ray is going forward and the exit point is after the enter point, compute the intersection point.
      intersection = p0 + t_exit * d
      return np.array(intersection)

# ------------------ OBJECT POSITION SAMPLING UTILS ------------------ #

def semicircle(radius, lower_bound=0, upper_bound=np.pi, n_points=200):
    """Return a circular sector polygon centred at the origin.

    Args:
        radius: Sector radius.
        lower_bound: Start angle, in radians.
        upper_bound: End angle, in radians.
        n_points: Number of points along the arc.
    """
    angles = [lower_bound + (upper_bound - lower_bound) * i / (n_points - 1) for i in range(n_points)]
    points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    points += [(0, 0)]  # close the shape
    return Polygon(points)

def c_shape(r_inner, r_outer, lower_bound=0, upper_bound=np.pi):
    """Return the C-shaped region between two radii over an angle range.

    Args:
        r_inner: Inner radius.
        r_outer: Outer radius.
        lower_bound: Start angle, in radians.
        upper_bound: End angle, in radians.
    """
    outer = semicircle(r_outer, lower_bound=lower_bound, upper_bound=upper_bound)
    inner = semicircle(r_inner, lower_bound=lower_bound, upper_bound=upper_bound)
    c_region = outer.difference(inner, grid_size=GRID_SIZE)
    
    return c_region

def sample_rectangle_in_cshape(c_region, forbidden, width, height, max_tries=10000):
    """Sample a randomly placed and rotated rectangle inside a region, avoiding a forbidden area.

    Args:
        c_region: Shapely region to sample inside.
        forbidden: Shapely geometry the rectangle must not touch.
        width: Rectangle width.
        height: Rectangle height.
        max_tries: Number of attempts before giving up.

    Returns:
        (rect, cx, cy, angle), or None if no sample fits.
    """
    minx, miny, maxx, maxy = c_region.bounds
    for _ in range(max_tries):
        cx = np.random.uniform(minx, maxx)
        cy = np.random.uniform(miny, maxy)
        angle = np.random.uniform(0, 2 * np.pi)
        
        rect = box(-width/2, -height/2, width/2, height/2)
        rect = rotate(rect, angle, use_radians=True)
        rect = translate(rect, cx, cy)
        
        # Check: fully inside C and no overlap with forbidden region
        if rect.within(c_region) and rect.intersects(forbidden) is False:
            return rect, cx, cy, angle
    return None  # no valid sample found

def sample_rectangle_in_axis_aligned_rect(rect_region, forbidden, width, height, max_tries=10000):
    """Sample an axis-aligned rectangle inside an axis-aligned region, avoiding a forbidden area.

    Args:
        rect_region: Axis-aligned shapely rectangle to sample inside.
        forbidden: Shapely geometry the rectangle must not touch (may be empty).
        width: Rectangle width.
        height: Rectangle height.
        max_tries: Number of attempts before giving up.

    Returns:
        (rect, cx, cy, 0.0), or None if no sample fits.
    """
    minx, miny, maxx, maxy = rect_region.bounds
    
    # Check if the rectangle can fit within the bounds
    if (maxx - minx) < width or (maxy - miny) < height:
        return None  # Rectangle is too large to fit
    
    for _ in range(max_tries):
        # Sample center position, ensuring the rectangle fits within bounds
        cx = np.random.uniform(minx + width/2, maxx - width/2)
        cy = np.random.uniform(miny + height/2, maxy - height/2)
        
        # Create axis-aligned rectangle (no rotation)
        rect = box(cx - width/2, cy - height/2, cx + width/2, cy + height/2)
        
        # Check: fully inside bounding rectangle and no overlap with forbidden region
        if rect.within(rect_region) and (forbidden.is_empty or rect.intersects(forbidden) is False):
            return rect, cx, cy, 0.0
    return None  # no valid sample found

def sample_object_pose_on_shelf(
    fixed_rect_center, # (x, y) tuple center of the fixed rectangle
    fixed_rect_dimensions, # (width, height) tuple dimensions of the fixed rectangle
    fixed_rect_angle, # Orientation angle in radians of the fixed rectangle
    sampled_rect_dimensions, # (width, height) tuple dimensions of the sampled rectangle
    max_tries=10000
    ):
    """Sample an object (x, y, yaw) whose footprint fits inside a rotated rectangle.

    Args:
        fixed_rect_center: (x, y) centre of the target rectangle.
        fixed_rect_dimensions: (width, height) of the target rectangle.
        fixed_rect_angle: Rotation of the target rectangle, in radians.
        sampled_rect_dimensions: (width, height) of the object footprint.
        max_tries: Number of attempts before giving up.

    Returns:
        (x, y, yaw), or None if the object does not fit.
    """
    fixed_rect = box(-fixed_rect_dimensions[0]/2, -fixed_rect_dimensions[1]/2, fixed_rect_dimensions[0]/2, fixed_rect_dimensions[1]/2)
    fixed_rect = rotate(fixed_rect, fixed_rect_angle, use_radians=True)
    fixed_rect = translate(fixed_rect, fixed_rect_center[0], fixed_rect_center[1])

    minx, miny, maxx, maxy = fixed_rect.bounds
    for _ in range(max_tries):
        cx = np.random.uniform(minx, maxx)
        cy = np.random.uniform(miny, maxy)
        angle = np.random.uniform(0, 2 * np.pi)
        
        rect = box(-sampled_rect_dimensions[0]/2, -sampled_rect_dimensions[1]/2, sampled_rect_dimensions[0]/2, sampled_rect_dimensions[1]/2)
        rect = rotate(rect, angle, use_radians=True)
        rect = translate(rect, cx, cy)
        
        # Check: fully inside C and no overlap with forbidden region
        if rect.within(fixed_rect):

            # fig, ax = plt.subplots()
            # for geom, color in [(fixed_rect, 'blue'), (rect, 'green')]:
            #     if geom:
            #         xs, ys = geom.exterior.xy
            #         ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')

            # ax.set_aspect('equal')
            # ax.set_xlim(0, 1)
            # ax.set_ylim(-1, 1)
            # plt.show()

            return cx, cy, angle

    print("Object is too large to fit inside the fixed rectangle")
    return None  # no valid sample found

def sample_point_in_fixed_rectangle(
    fixed_rect_center,
    fixed_rect_dimensions,
    fixed_rect_angle,
    num_points=1,
):
    """Sample random points uniformly inside a rotated rectangle.

    Args:
        fixed_rect_center: (x, y) centre of the rectangle.
        fixed_rect_dimensions: (width, height) of the rectangle.
        fixed_rect_angle: Rotation of the rectangle, in radians.
        num_points: Number of points to sample.

    Returns:
        An (x, y) tuple if num_points is 1, else an (N, 2) array.
    """
    half_w = fixed_rect_dimensions[0] * 0.5
    half_h = fixed_rect_dimensions[1] * 0.5

    # Sample in the rectangle's local frame
    dx = np.random.uniform(-half_w, half_w, size=num_points)
    dy = np.random.uniform(-half_h, half_h, size=num_points)

    # Rotate by fixed_rect_angle and translate to world frame
    c = np.cos(fixed_rect_angle)
    s = np.sin(fixed_rect_angle)
    x = fixed_rect_center[0] + c * dx - s * dy
    y = fixed_rect_center[1] + s * dx + c * dy

    if num_points == 1:
        return (x[0], y[0])
    else:
      return np.column_stack([x, y])

def sample_point_in_fixed_rectangle_uniformly(
    fixed_rect_center,
    fixed_rect_dimensions,
    fixed_rect_angle,
    grid_size = 10,
):
    """Return an evenly spaced grid of points covering a rotated rectangle.

    Args:
        fixed_rect_center: (x, y) centre of the rectangle.
        fixed_rect_dimensions: (width, height) of the rectangle.
        fixed_rect_angle: Rotation of the rectangle, in radians.
        grid_size: Number of points per side.

    Returns:
        (grid_size**2, 2) array of (x, y) points.
    """
    half_w = fixed_rect_dimensions[0] * 0.5
    half_h = fixed_rect_dimensions[1] * 0.5

    # Sample in the rectangle's local frame
    dx = np.linspace(-half_w, half_w, grid_size)
    dy = np.linspace(-half_h, half_h, grid_size)

    x, y = np.meshgrid(dx, dy)

    # Rotate by fixed_rect_angle and translate to world frame
    c = np.cos(fixed_rect_angle)
    s = np.sin(fixed_rect_angle)
    x = fixed_rect_center[0] + c * x - s * y
    y = fixed_rect_center[1] + s * x + c * y

    x = x.flatten()
    y = y.flatten()

    return np.column_stack([x, y])

def sample_object_pose_on_table(
    r_min, # Minimum radius of c-space semicircle
    r_max, # Maximum radius of c-space semicircle
    lower_angle, # Lower angle of c-space semicircle
    upper_angle, # Upper angle of c-space semicircle
    fixed_rect_center, # (x, y) tuple center of the fixed rectangle
    fixed_rect_dimensions, # (width, height) tuple dimensions of the fixed rectangle
    fixed_rect_angle, # Orientation angle in radians of the fixed rectangle
    sampled_rect_dimensions # (width, height) tuple dimensions of the sampled rectangle
    ):
    """Sample an object (x, y, yaw) in a C-shaped table region that avoids a fixed rectangle.

    Args:
        r_min: Inner radius of the C-shaped region.
        r_max: Outer radius of the C-shaped region.
        lower_angle: Start angle of the region, in radians.
        upper_angle: End angle of the region, in radians.
        fixed_rect_center: (x, y) centre of the rectangle to avoid.
        fixed_rect_dimensions: (width, height) of the rectangle to avoid.
        fixed_rect_angle: Rotation of the rectangle to avoid, in radians.
        sampled_rect_dimensions: (width, height) of the object footprint.
    """
    
    fixed_rect = box(-fixed_rect_dimensions[0]/2, -fixed_rect_dimensions[1]/2, fixed_rect_dimensions[0]/2, fixed_rect_dimensions[1]/2)
    fixed_rect = rotate(fixed_rect, fixed_rect_angle, use_radians=True)
    fixed_rect = translate(fixed_rect, fixed_rect_center[0], fixed_rect_center[1])

    c_region = c_shape(r_min, r_max, lower_bound=lower_angle, upper_bound=upper_angle)

    forbidden = fixed_rect.intersection(c_region, grid_size=GRID_SIZE)

    rect_sampled, cx, cy, angle = sample_rectangle_in_cshape(c_region, forbidden, width=sampled_rect_dimensions[0], height=sampled_rect_dimensions[1])

    # fig, ax = plt.subplots()
    # for geom, color in [(c_region, 'lightgray'), (fixed_rect, 'blue'), (rect_sampled, 'green')]:
    #     if geom:
    #         xs, ys = geom.exterior.xy
    #         ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')

    # ax.set_aspect('equal')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-1, 1)
    # plt.show()

    return cx, cy, angle


def sample_object_pose_on_table_multi_object(
    x_min, # Minimum X bound of the sampling rectangle
    x_max, # Maximum X bound of the sampling rectangle
    y_min, # Minimum Y bound of the sampling rectangle
    y_max, # Maximum Y bound of the sampling rectangle
    fixed_rect_centers, # List of (x, y) tuple centers of the fixed rectangles
    fixed_rect_dimensions, # List of (width, height) tuple dimensions of the fixed rectangles
    fixed_rect_angles, # List of orientation angles in radians of the fixed rectangles
    sampled_rect_dimensions # (width, height) tuple dimensions of the sampled rectangle
    ):
    """Sample an axis-aligned object position in a box that avoids several fixed rectangles.

    Args:
        x_min: Lower x bound of the sampling box.
        x_max: Upper x bound of the sampling box.
        y_min: Lower y bound of the sampling box.
        y_max: Upper y bound of the sampling box.
        fixed_rect_centers: List of (x, y) centres of the rectangles to avoid.
        fixed_rect_dimensions: List of (width, height) of the rectangles to avoid.
        fixed_rect_angles: List of rotations of the rectangles to avoid, in radians.
        sampled_rect_dimensions: (width, height) of the object footprint.

    Returns:
        (x, y, 0.0), or None if no position fits.
    """
    
    # Create axis-aligned bounding rectangle
    rect_region = box(x_min, y_min, x_max, y_max)
    
    # Create all fixed rectangles and compute their intersections with rect_region
    forbidden_regions = []
    for center, dimensions, angle in zip(fixed_rect_centers, fixed_rect_dimensions, fixed_rect_angles):
        fixed_rect = box(-dimensions[0]/2, -dimensions[1]/2, dimensions[0]/2, dimensions[1]/2)
        fixed_rect = rotate(fixed_rect, angle, use_radians=True)
        fixed_rect = translate(fixed_rect, center[0], center[1])
        
        # Get intersection of this fixed rectangle with rect_region
        forbidden_region = fixed_rect.intersection(rect_region, grid_size=GRID_SIZE)
        if forbidden_region and not forbidden_region.is_empty:
            forbidden_regions.append(forbidden_region)
    
    # Combine all forbidden regions into a single geometry
    if forbidden_regions:
        forbidden = unary_union(forbidden_regions)
    else:
        # If no forbidden regions, create an empty polygon
        forbidden = Polygon()

    result = sample_rectangle_in_axis_aligned_rect(rect_region, forbidden, width=sampled_rect_dimensions[0], height=sampled_rect_dimensions[1])
    if result is None:
        print("Failed to sample rectangle within bounds")
        return None
    
    rect_sampled, cx, cy, angle = result

    # visualize_regions([(rect_region, 'lightgray'), (forbidden, 'red'), (rect_sampled, 'green')])

    return cx, cy, angle

def visualize_regions(regions):
  """Plot shapely regions with matplotlib.

  Args:
    regions: List of (geometry, color) pairs.
  """
  fig, ax = plt.subplots()
  for geom, color in regions:
      if geom:
          # Handle MultiPolygon by iterating over constituent polygons
          if isinstance(geom, MultiPolygon):
              for poly in geom.geoms:
                  xs, ys = poly.exterior.xy
                  ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')
          elif isinstance(geom, Polygon):
              xs, ys = geom.exterior.xy
              ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')
          else:
              # For other geometry types, try to get exterior if available
              if hasattr(geom, 'exterior'):
                  xs, ys = geom.exterior.xy
                  ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')
  ax.set_aspect('equal')
  ax.set_xlim(0, 1)
  ax.set_ylim(-1, 1)
  plt.show()
    

# if __name__ == "__main__":
#     # Define semicircles and rectangle
#     center = (0, 0)
#     r_inner, r_outer = 3.0, 5.0
#     y_limit = 1.0  # C-region starts only from y=1.0
#     c_region = c_shape(center, r_inner, r_outer, lower_bound=0, upper_bound=np.pi, y_limit=y_limit)

#     # Example fixed rectangle
#     rect_existing = box(-1, 0, 1, 1)    # minx, miny, maxx, maxy
#     forbidden = rect_existing.intersection(c_region)

#     # Sample a new rectangle
#     rect_sampled = sample_rectangle_in_cshape(c_region, forbidden, width=0.5, height=0.5)

#     print(rect_sampled)
    
#     fig, ax = plt.subplots()
#     for geom, color in [(c_region, 'lightgray'), (rect_existing, 'blue'), (rect_sampled, 'green')]:
#         if geom:
#             xs, ys = geom.exterior.xy
#             ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')

#     ax.set_aspect('equal')
#     plt.show()