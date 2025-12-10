from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
import os

import time
from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Concatenate,
    Context,
    Diagram,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    PiecewisePolynomial,
    PiecewisePose,
    PointCloud,
    Rgba,
    RigidTransform,
    RobotDiagram,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    Trajectory,
    TrajectorySource,
)

from manipulation.icp import IterativeClosestPoint

from pydrake.systems.primitives import ConstantVectorSource
import numpy as np
from pydrake.geometry import Box, Rgba, Sphere
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from manipulation import running_as_notebook

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    AddPointClouds
)

from manipulation.utils import RenderDiagram
from pydrake.geometry import Box, Rgba, Sphere
from pydrake.math import RigidTransform, RotationMatrix

from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import Diagram

# Start meshcat for visualization
meshcat = StartMeshcat()
print("Click the link above to open Meshcat in your browser!")

table_sdf = """
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="table">
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.0333</ixx>
          <iyy>0.0333</iyy>
          <izz>0.005</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>2 2 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>2 2 0.1</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
os.makedirs("assets", exist_ok=True)

with open("assets/table.sdf", "w") as f:
    f.write(table_sdf)

# Add the directives for the bimanual IIWA arms, table, and initials


def generate_bimanual_IIWA14_with_assets_directives_file() -> (
    tuple[Diagram, RobotDiagram]
):
    table_sdf = f"{Path.cwd()}/assets/table.sdf"

    directives_yaml = f"""directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
    X_PC:
        translation: [0, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
- add_model:
    name: table
    file: file://{table_sdf}
- add_weld:
    parent: world
    child: table::link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
"""
    os.makedirs("directives", exist_ok=True)

    with open(
        "directives/bimanual_IIWA14_with_table_and_initials_and_assets.dmd.yaml", "w"
    ) as f:
        f.write(directives_yaml)

# - add_model:
#     name: table
#     file: file://{table_sdf}
# - add_weld:
#     parent: world
#     child: table::link
#     X_PC:
#         translation: [0.0, 0.0, -0.05]
#         rotation: !Rpy {{ deg: [0, 0, -90] }}
generate_bimanual_IIWA14_with_assets_directives_file()

def create_camera_directives() -> None:
    """
    Define a single hand-mounted camera (camera0) attached to the WSG body.

    The frame `camera0_origin` is attached to `wsg::body`, and the camera
    model is welded to that frame. This matches your requested:
    """
    camera_directives_yaml = """
directives:
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: wsg::body
        rotation: !Rpy { deg: [-90.0, 0.0, 0.0]}
        translation: [0.0, 0.0, 0.05]

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: wsg::camera0_origin
    child: camera0::base
"""
    os.makedirs("directives", exist_ok=True)
    with open("directives/camera_directives.dmd.yaml", "w") as f:
        f.write(camera_directives_yaml)



create_camera_directives()

def create_cabinet_directives() -> None:
    cabinet_directives_yaml = f"""
directives:
- add_model:
    name: drawer
    file: file:///{Path.cwd()}/assets/drawer.urdf
    default_joint_positions:
        base_drawer_joint: [0.0]
- add_weld:
    parent: world
    child: drawer::base_link
    X_PC:
        translation: [0.0, 1.0, 0.2]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
"""
    os.makedirs("directives", exist_ok=True)

    with open(
        "directives/cabinet_directives.dmd.yaml", "w"
    ) as f:
        f.write(cabinet_directives_yaml)

create_cabinet_directives()

def create_bimanual_IIWA14_with_assets_and_cameras_scenario() -> None:
    # TODO: create a scenario yaml with the directives added with `add_directives`
    scenario_yaml = f"""
directives:
    - add_directives:
        file: file://{Path.cwd()}/directives/bimanual_IIWA14_with_table_and_initials_and_assets.dmd.yaml
    - add_directives:
        file: "file://{Path.cwd()}/directives/camera_directives.dmd.yaml"
    - add_directives:
        file: "file://{Path.cwd()}/directives/cabinet_directives.dmd.yaml"


cameras:
    camera0:
        name: camera0
        depth: True
        X_PB:
            base_frame: camera0::base

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_and_torque
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""


    # TODO: add the camera configs and iiwa drivers with `add_cameras` and `add_iiwa_drivers`

    os.makedirs("scenarios", exist_ok=True)

    with open(
        "scenarios/bimanual_IIWA14_with_table_and_initials_and_assets_and_cameras.scenario.yaml",
        "w",
    ) as f:
        f.write(scenario_yaml)


create_bimanual_IIWA14_with_assets_and_cameras_scenario()

def create_bimanual_IIWA14_with_table_and_initials_and_assets_and_cameras() -> (
    tuple[DiagramBuilder, RobotDiagram]
):
    # TODO: Load the scenario created above into a Scenario object
    scenario_path = "./scenarios/bimanual_IIWA14_with_table_and_initials_and_assets_and_cameras.scenario.yaml"
    scenario = LoadScenario(filename = scenario_path)
    # TODO: Create HardwareStation with the scenario and meshcat
    station = MakeHardwareStation(scenario, meshcat = meshcat)
    # TODO: Make a DiagramBuilder, add the station, and build the diagram
    builder = DiagramBuilder()
    station_sys = builder.AddSystem(station)
    # TODO: Add the point clouds to the diagram with AddPointClouds
    pc_ports = AddPointClouds(
        scenario = scenario,
        builder=builder,
        station=station_sys,
        meshcat=meshcat)
    # TODO: export the point cloud outputs to the builder
    for name, to_pc_sys in pc_ports.items():
        builder.ExportOutput(to_pc_sys.point_cloud_output_port(), f"{name}_point_cloud")
    # TODO: Return the builder AND the station (notice that here we will need both)
    return builder, station

from dataclasses import dataclass

@dataclass
class Plane:
    p0: np.ndarray  # point on plane (3,)
    n: np.ndarray   # unit normal (3,)

def fit_plane_ransac(
    points: np.ndarray,
    num_iters: int = 500,
    distance_thresh: float = 0.005,  # 5 mm inliers for plane
    min_inliers: int = 1000,
) -> Plane:
    """
    Simple RANSAC plane fit.
    points: (N, 3) numpy array in some frame (e.g., world).
    Returns Plane(p0, n) in that same frame.
    """
    N = points.shape[0]
    best_inliers = -1
    best_p0 = None
    best_n = None

    if N < 3:
        raise RuntimeError("Not enough points to fit a plane.")

    for _ in range(num_iters):
        # Randomly pick 3 distinct points
        idxs = np.random.choice(N, 3, replace=False)
        a, b, c = points[idxs]

        # Normal from cross product
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            continue  # degenerate triple

        n = n / norm_n

        # Distances of all points to this plane
        d = np.abs((points - a) @ n)  # |n · (p - a)|
        inliers = np.count_nonzero(d < distance_thresh)

        if inliers > best_inliers:
            best_inliers = inliers
            best_p0 = a
            best_n = n

    if best_inliers < min_inliers:
        print(f"Warning: RANSAC found only {best_inliers} inliers (< {min_inliers}).")

    print(f"RANSAC plane: {best_inliers} inliers, p0 = {best_p0}, n = {best_n}")
    return Plane(p0=best_p0, n=best_n)

def segment_handle_from_camera0(
    diagram,
    context,
    distance_thresh: float = 0.06,  # points closer than this to plane are removed
):
    """
    Uses RANSAC on camera0_point_cloud to find the drawer plane,
    then removes all points within distance_thresh of that plane.
    Returns (plane, handle_points) where handle_points is (M, 3).
    """
    # Get camera0 point cloud from diagram
    pc_port = diagram.GetOutputPort("camera0_point_cloud")
    pc: PointCloud = pc_port.Eval(context)

    # Drake PointCloud stores xyzs as (3, N)
    xyzs = pc.xyzs()
    points = np.asarray(xyzs.T)  # (N, 3)

    mask = points[:, 2] > 0.1     # boolean mask for z coordinate
    points = points[mask]

    # 1) Fit plane with RANSAC
    plane = fit_plane_ransac(
        points,
        num_iters=500,
        distance_thresh=0.005,  # inlier threshold for plane fit
        min_inliers=2000,
    )

    # 2) Compute point–plane distances and filter
    # distance = | n · (p - p0) |
    dists = np.abs((points - plane.p0) @ plane.n)
    mask_handle = dists > distance_thresh  # keep points far from plane
    handle_points = points[mask_handle]

    print(f"Total points: {points.shape[0]}")
    print(f"Plane inliers removed: {(~mask_handle).sum()}")
    print(f"Handle/remaining points: {handle_points.shape[0]}")

    return plane, handle_points

def visualize_handle_points(meshcat, handle_points: np.ndarray, path="handle_points"):
    """
    Visualize the handle-only point cloud (points not near the drawer plane)
    as a point cloud in MeshCat.
    handle_points: (M, 3) array in the same frame as the original cloud.
    """
    if handle_points.shape[0] == 0:
        print("No handle points to visualize.")
        return

    # PointCloud with XYZ only (no RGBs)
    pc_handle = PointCloud(handle_points.shape[0])
    pc_handle.mutable_xyzs()[:] = handle_points.T  # (3, M)

    # Give MeshCat an overall color and point size
    meshcat.SetObject(
        path,
        pc_handle,
        point_size=0.003,                    # tweak if it's too small / big
        rgba=Rgba(1.0, 0.0, 0.0, 1.0),       # red handle points
    )


from typing import Tuple, List, Optional

# (p1, p2, n1, n2) in the SAME frame (we'll use world frame W == object frame O)
AntipodeCandidateType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def compute_grasp_from_points(
    antipodal_pt: AntipodeCandidateType,
) -> Optional[RigidTransform]:
    """
    Given the tuple of antipodal points and their normals on the object O,
    compute the grasp X_OG.

    Convention:
      - Object frame O has z-axis up: z_O = [0, 0, 1].
      - Gripper frame G:
          x_G: points along surface normal (n1).
          y_G: points DOWN (-z_O), projected to be orthogonal to x_G.
          z_G: completes the right-handed frame.
      - The grasp origin is at the midpoint of the two contact points.
      - We then add an offset of -0.1m along y_G to account for finger length
        (so the fingers end up around the contact points).
    """
    print("Computing grasp from points...")
    z_axis_O = np.array([0.0, 0.0, 1.0])

    p1, p2, n1, n2 = antipodal_pt

    # 1) x-axis of G is n1 (surface normal at first contact), normalized
    x_axis = np.array(n1, dtype=float)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        return None
    x_axis = x_axis / x_norm

    # 2) If x-axis is nearly parallel to z_O, we skip (bad for top-down grasp construction)
    if abs(np.dot(x_axis, z_axis_O)) > 0.99:
        return None

    # 3) y-axis of G should point "down" (towards -z_O) but be orthogonal to x_G.
    #    Use Gram-Schmidt projection of -z_O onto the plane orthogonal to x_G.
    y_desired = -z_axis_O
    y_axis = y_desired - np.dot(y_desired, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        return None
    y_axis = y_axis / y_norm

    # 4) z-axis of G completes a right-handed frame
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        return None
    z_axis = z_axis / z_norm

    # 5) Construct rotation R_OG
    #    Columns are [x_G, y_G, z_G] expressed in O.
    R_OG_mat = np.column_stack((x_axis, y_axis, z_axis))
    R_OG = RotationMatrix(R_OG_mat)

    # 6) Grasp position p_OG_O: midpoint of the two contact points (in O)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p_OG_O = 0.5 * (np.array(p1, dtype=float) + np.array(p2, dtype=float))
    X_OG = RigidTransform(R_OG, p_OG_O)
    X_OG_offset = X_OG @ RigidTransform([-0.1, 0.0, 0])
    return X_OG_offset


def check_collision_free(X_WG: RigidTransform) -> bool:
    """
    Checks if the gripper collides with the table.

    - Table is modeled as plane z=0 in world frame.
    - Gripper is approximated as a small box with two key vertices
      (roughly the "lowest" parts of the fingers/palm).
    """
    # normalize input to a 4x4 numpy matrix
    if hasattr(X_WG, "GetAsMatrix4"):
        X_mat = np.array(X_WG.GetAsMatrix4(), dtype=float)
    elif hasattr(X_WG, "matrix"):
        X_mat = np.array(X_WG.matrix(), dtype=float)
    else:
        X_mat = np.array(X_WG, dtype=float)

    # ensure we have a 4x4 homogeneous transform
    if X_mat.shape == (3, 4):
        X_mat = np.vstack((X_mat, [0.0, 0.0, 0.0, 1.0]))
    elif X_mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform or compatible object, got shape {X_mat.shape}")

    # (very crude) vertices modeling the gripper collision body in G
    # You can tweak these if you know your WSG geometry better.
    gripper_vertices_G = np.array([
        [-0.073, -0.085383, -0.025],
        [ 0.073,  0.069,     0.025],
    ])
    verts_h = np.hstack((gripper_vertices_G, np.ones((gripper_vertices_G.shape[0], 1))))

    # map to world
    world_verts_h = (X_mat @ verts_h.T).T
    z_vals = world_verts_h[:, 2]

    # collision-free if *all* vertices are above z=0
    return bool(np.all(z_vals > 0.0 + 1e-8))


def get_filtered_grasps(
    candidate_list: List[AntipodeCandidateType],
    antipodal_thresh: float,
    z_axis_thresh: float,
    max_pt_dist: float,
    min_pt_dist: float,
    X_WO: RigidTransform,
    require_top_down: bool = False,
    check_collisions: bool = False,
) -> List[RigidTransform]:
    """
    Filter a list of antipodal candidates into valid grasps in WORLD frame.

    Filters:
      (1) Antipodality: normals roughly opposite.
      (2) Point distance: within [min_pt_dist, max_pt_dist].
      (3) Optional top-down approach: gripper y-axis points roughly downward.
      (4) Optional collision-free wrt table z=0.

    Inputs:
        candidate_list: list[(p1, p2, n1, n2)] in object frame O.
        X_WO: RigidTransform from object frame O to world frame W.
    Returns:
        grasps_W: list of RigidTransform X_WG (gripper in world frame).
    """
    z_world = np.array([0.0, 0.0, 1.0])
    filtered_candidates: List[RigidTransform] = []

    for candidate in candidate_list:
        # p1, p2, n1, n2 = candidate

        # # 1) Antipodality: normals should be roughly opposite, so dot(n1, n2) < threshold
        # n1_u = np.array(n1, dtype=float)
        # n2_u = np.array(n2, dtype=float)
        # n1_u = n1_u / (np.linalg.norm(n1_u) + 1e-12)
        # n2_u = n2_u / (np.linalg.norm(n2_u) + 1e-12)
        # normals_dot = float(np.dot(n1_u, n2_u))
        # if normals_dot >= antipodal_thresh:
        #     # e.g. threshold = -0.9 → require angle > ~154°
        #     continue

        # # 2) Point distance filter
        # pt_dist = float(np.linalg.norm(
        #     np.array(p1, dtype=float) - np.array(p2, dtype=float)
        # ))
        # if pt_dist > max_pt_dist or pt_dist < min_pt_dist:
        #     continue

        # # 3) Compute grasp in object frame O
        X_OG = compute_grasp_from_points(candidate)
        if X_OG is None:
            continue

        # 4) Map to world frame: X_WG = X_WO * X_OG
        X_WG = X_WO @ X_OG

        # # 5) Optional top-down constraint:
        # #    We want gripper y-axis to point roughly along -z_world.
        # if require_top_down:
        #     try:
        #         R_WG = np.array(X_WG.rotation().matrix())
        #     except Exception:
        #         X_mat = np.array(X_WG.GetAsMatrix4(), dtype=float)
        #         R_WG = X_mat[:3, :3]

        #     y_world = R_WG[:, 1]   # y-axis of G in world frame
        #     # Require y_world·z_world <= -z_axis_thresh (i.e. sufficiently downwards)
        #     if y_world[2] > -z_axis_thresh:
        #         continue

        # # 6) Optional collision check against table
        # if check_collisions and not check_collision_free(X_WG):
        #     continue

        filtered_candidates.append(X_WG)

    return filtered_candidates


AntipodeCandidateType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (p1, p2, n1, n2)


# ----------------------------------------------------------------------
# 1. Normal estimation on a point cloud using local PCA
# ----------------------------------------------------------------------
def estimate_normals_pca(
    points: np.ndarray,
    k_neighbors: int = 30,
) -> np.ndarray:
    """
    Estimate surface normals for each point in a point cloud using PCA
    of the k-nearest neighbors.

    points: (N, 3) array in some frame (we'll treat it as world frame W).
    Returns normals: (N, 3) array, unit-length, roughly pointing outward.
    """
    N = points.shape[0]
    if N < k_neighbors + 1:
        raise RuntimeError(f"Not enough points ({N}) for k_neighbors={k_neighbors}.")

    normals = np.zeros_like(points)
    # global center to orient normals consistently
    center = points.mean(axis=0)

    # brute-force neighbor search (OK for a few thousand points)
    for i in range(N):
        p = points[i]
        dists = np.linalg.norm(points - p, axis=1)
        # Exclude the point itself, then pick the nearest k_neighbors
        idx = np.argsort(dists)[1 : k_neighbors + 1]
        neighborhood = points[idx] - p  # (k, 3)

        # PCA: smallest eigenvector of covariance is normal
        cov = neighborhood.T @ neighborhood
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, 0]  # eigenvector with smallest eigenvalue

        # Orient normal to point roughly away from the centroid
        if np.dot(n, p - center) < 0.0:
            n = -n

        n /= (np.linalg.norm(n) + 1e-12)
        normals[i] = n

    return normals


# ----------------------------------------------------------------------
# 2. Sample antipodal candidates directly from point cloud + normals
# ----------------------------------------------------------------------
def sample_antipodal_candidates_from_point_cloud(
    points: np.ndarray,
    normals: np.ndarray,
    n_candidates: int = 2000,
    max_pt_dist: float = 0.08,
    min_pt_dist: float = 0.01,
) -> List[AntipodeCandidateType]:
    """
    Build candidate antipodal point pairs directly from a point cloud.

    This version avoids building an O(N^2) distance matrix. Instead, it
    randomly samples point pairs and keeps the ones whose distance lies in
    [min_pt_dist, max_pt_dist].

    points:   (N, 3)
    normals:  (N, 3) estimated normals at each point
    n_candidates: approximate number of point pairs to sample
    max_pt_dist, min_pt_dist: rough distance band for candidate creation
    """
    N = points.shape[0]
    if N < 2:
        return []

    candidates: List[AntipodeCandidateType] = []

    # Allow some slack if pairs fail distance threshold
    max_trials = n_candidates * 50

    for _ in range(max_trials):
        i = np.random.randint(N)
        j = np.random.randint(N - 1)
        if j >= i:
            j += 1  # ensure j != i

        p1 = points[i]
        p2 = points[j]
        d = np.linalg.norm(p1 - p2)

        if d < min_pt_dist or d > max_pt_dist:
            continue

        n1 = normals[i]
        n2 = normals[j]

        candidates.append((p1, p2, n1, n2))

        if len(candidates) >= n_candidates:
            break

    print(f"[sample_antipodal_candidates_from_point_cloud] built "
          f"{len(candidates)} candidates (requested {n_candidates})")
    return candidates



# ----------------------------------------------------------------------
# 3. Full pipeline: camera0 -> handle points -> antipodal grasps in W
# ----------------------------------------------------------------------
def find_filtered_grasps_from_camera0(
    diagram,
    context,
    n_normal_neighbors: int = 30,
    n_antipodal_candidates: int = 3000,
    plane_fit_distance_thresh: float = 0.005,
    plane_fit_min_inliers: int = 2000,
    handle_plane_reject_thresh: float = 0.06,
    antipodal_thresh: float = -0.9,
    z_axis_thresh: float = 0.5,
    max_pt_dist: float = 0.08,
    min_pt_dist: float = 0.01,
    require_top_down: bool = False,
    check_collisions: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[RigidTransform]]:
    """
    1) Use camera0 point cloud to segment a drawer plane and keep only
       'handle' points.
    2) Estimate normals on handle points.
    3) Sample antipodal candidates from the handle point cloud.
    4) Filter them using get_filtered_grasps, treating object frame O
       as identical to world frame W (X_WO = Identity).

    Returns:
        handle_points: (M, 3) array of the handle-only cloud
        grasps_W:      list of RigidTransform (X_WG) in world frame
    """
    # 1) segment handle from camera0
    plane, handle_points = segment_handle_from_camera0(
        diagram,
        context,
        distance_thresh=handle_plane_reject_thresh,
    )
    if handle_points.shape[0] < 10:
        print(f"[find_filtered_grasps_from_camera0] Too few handle points: {handle_points.shape[0]}")
        return handle_points, []

    if verbose:
        print(f"[find_filtered_grasps_from_camera0] Got {handle_points.shape[0]} handle points")

    # 2) estimate normals on handle points in whatever frame they're in
    normals = estimate_normals_pca(handle_points, k_neighbors=n_normal_neighbors)
    if verbose:
        print("[find_filtered_grasps_from_camera0] Estimated normals via PCA")

    # 3) sample antipodal candidates directly from handle cloud
    candidates = sample_antipodal_candidates_from_point_cloud(
        handle_points,
        normals,
        n_candidates=n_antipodal_candidates,
        max_pt_dist=max_pt_dist,
        min_pt_dist=min_pt_dist,
    )
    if verbose:
        print(f"[find_filtered_grasps_from_camera0] Sampled {len(candidates)} antipodal candidates")

    # 4) Filter candidates into valid grasps.
    #    We treat the "object frame" O as the same as the world frame W,
    #    so X_WO = Identity.
    X_WO = RigidTransform()  # identity

    grasps_W = get_filtered_grasps(
        candidate_list=candidates,
        antipodal_thresh=antipodal_thresh,
        z_axis_thresh=z_axis_thresh,
        max_pt_dist=max_pt_dist,
        min_pt_dist=min_pt_dist,
        X_WO=X_WO,
        require_top_down=require_top_down,
        check_collisions=check_collisions,
    )

    if verbose:
        print(f"[find_filtered_grasps_from_camera0] Filtered down to {len(grasps_W)} valid grasps")

    return handle_points, grasps_W


from manipulation.meshcat_utils import AddMeshcatTriad


# ============================================================
# MeshCat triad / grasp-frame visualization
# ============================================================

def visualize_grasp_frames(
    meshcat,
    grasps_W,
    max_to_show=10,
    base_path="grasp_candidates",
    axis_length=0.05,
    axis_radius=0.002,
):
    num_to_show = min(len(grasps_W), max_to_show)
    for i, X_WG in enumerate(grasps_W[:num_to_show]):
        AddMeshcatTriad(
            meshcat,
            path=f"{base_path}/grasp_{i}",
            X_PT=X_WG,         # pose to draw at
            length=axis_length,
            radius=axis_radius,
        )
    print(f"Visualized {num_to_show} grasp frames.")

from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

def move_wsg_camera_to_initial_pose(
    diagram: Diagram,
    station: Diagram,
    context: Context,
    position_tolerance: float = 0.01,
    orientation_tolerance_deg: float = 0.0,
) -> None:
    """
    Uses IK to move iiwa so that the hand-mounted camera ends up at the
    *old* static camera0 pose in world, enforcing both position and
    (approximately) orientation of the WSG body.

    Old static camera pose (before hand-mounting):
      R_WC_old = Rpy(-90°, 0, 0)
      p_WC_old = [0.0, -0.3, 0.2]

    Hand-mounted transform (from directives):
      X_GC: Rpy(-90°, 0, 0), p_GC = [0, 0, 0.05]

    We solve for X_WG_target = X_WC_target * X_GC⁻¹ and run IK so that
    wsg::body ≈ X_WG_target (position + orientation).
    """
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    W = plant.world_frame()
    wsg_body = plant.GetBodyByName("body")
    wsg_frame = wsg_body.body_frame()

    # --- Old static camera pose in world ---
    R_WC_old = RotationMatrix.MakeXRotation(-math.pi / 2.0)  # -90 deg about x
    p_WC_old = np.array([0.0, 0.0, 0.2])
    X_WC_target = RigidTransform(R_WC_old, p_WC_old)

    # --- Hand-to-camera transform from directives ---
    R_GC = RotationMatrix.MakeXRotation(-math.pi / 2.0)
    p_GC = np.array([0.0, 0.0, 0.05])
    X_GC = RigidTransform(R_GC, p_GC)

    # Desired wsg pose so that camera ends up at old static pose
    X_WG_target = X_WC_target @ X_GC.inverse()
    R_WG_target = X_WG_target.rotation()
    p_WG_target = X_WG_target.translation()

    # --- Set up IK ---
    ik = InverseKinematics(plant, plant_context)
    q_decision = ik.q()

    # 1) Position constraint on WSG origin
    p_BQ = np.array([0.0, 0.0, 0.0])
    lower = p_WG_target - position_tolerance
    upper = p_WG_target + position_tolerance
    ik.AddPositionConstraint(
        frameB=wsg_frame,
        p_BQ=p_BQ,
        frameA=W,
        p_AQ_lower=lower,
        p_AQ_upper=upper,
    )

    # 2) Orientation constraint: wsg::body ≈ R_WG_target within theta_bound
    theta_bound = math.radians(orientation_tolerance_deg)
    # Correct signature / kw names:
    # AddOrientationConstraint(frameAbar, R_AbarA, frameBbar, R_BbarB, theta_bound)
    ik.AddOrientationConstraint(
        frameAbar=W,
        R_AbarA=RotationMatrix(),   # identity
        frameBbar=wsg_frame,
        R_BbarB=R_WG_target,
        theta_bound=theta_bound,
    )

    prog = ik.prog()
    # Initial guess: current plant positions
    q0 = plant.GetPositions(plant_context)
    prog.SetInitialGuess(q_decision, q0)

    result = Solve(prog)
    if not result.is_success():
        print(
            "[move_wsg_camera_to_initial_pose] IK failed with "
            f"pos_tol={position_tolerance}, ori_tol={orientation_tolerance_deg} deg; "
            "using default configuration."
        )
        return

    q_sol = result.GetSolution(q_decision)
    plant.SetPositions(plant_context, q_sol)

    print(
        "[move_wsg_camera_to_initial_pose] IK success; moved hand toward old camera pose "
        f"(pos tol={position_tolerance} m, ori tol={orientation_tolerance_deg} deg)."
    )


def compute_wsg_camera_target_pose() -> RigidTransform:
    """
    Returns X_WG_target: the desired wsg::body pose that makes the
    hand-mounted camera coincide with the 'old' static camera pose.

    This matches the math used in move_wsg_camera_to_initial_pose().
    """
    # Old static camera pose in world
    R_WC_old = RotationMatrix.MakeXRotation(-math.pi / 2.0)  # -90 deg about x
    p_WC_old = np.array([0.0, 0.0, 0.2])
    X_WC_target = RigidTransform(R_WC_old, p_WC_old)

    # Hand-to-camera transform from directives
    R_GC = RotationMatrix.MakeXRotation(-math.pi / 2.0)
    p_GC = np.array([0.0, 0.0, 0.05])
    X_GC = RigidTransform(R_GC, p_GC)

    # Desired wsg pose so that camera ends up at old static pose
    X_WG_target = X_WC_target @ X_GC.inverse()
    return X_WG_target


# ============================================================
# Build diagram, run the pipeline, and visualize
# ============================================================
def build_diagram_and_context():
    """
    Build the station + sensors Diagram, then:
      1) Solve IK to move the hand-mounted camera to (approximately) the
         old static camera pose.
      2) Publish once so camera0_point_cloud reflects that configuration.
    """
    builder, station = create_bimanual_IIWA14_with_table_and_initials_and_assets_and_cameras()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Move the arm so that the hand-mounted camera mimics the old
    # static camera0 viewpoint before we grab the point cloud.
    move_wsg_camera_to_initial_pose(diagram, station, context)

    # Render sensors (incl. camera0 point cloud) once for this configuration
    diagram.ForcedPublish(context)

    return diagram, context, station



def run_handle_grasp_detection_and_visualization(
    n_normal_neighbors: int = 30,
    n_antipodal_candidates: int = 3000,
    antipodal_thresh: float = -0.9,
    z_axis_thresh: float = 0.5,
    max_pt_dist: float = 0.08,
    min_pt_dist: float = 0.01,
    require_top_down: bool = False,
    check_collisions: bool = False,
    max_grasps_to_show: int = 10,
):
    """
    1) Builds the diagram + context.
    2) Segments handle points from camera0 using RANSAC plane.
    3) Finds filtered antipodal grasps with find_filtered_grasps_from_camera0().
    4) Visualizes handle point cloud and grasp frames in MeshCat.
    """
    diagram, context, station = build_diagram_and_context()

    # --- Run point cloud → handle segmentation → antipodal grasps ---
    handle_points, grasps_W = find_filtered_grasps_from_camera0(
        diagram,
        context,
        n_normal_neighbors=n_normal_neighbors,
        n_antipodal_candidates=n_antipodal_candidates,
        antipodal_thresh=antipodal_thresh,
        z_axis_thresh=z_axis_thresh,
        max_pt_dist=max_pt_dist,
        min_pt_dist=min_pt_dist,
        require_top_down=require_top_down,
        check_collisions=check_collisions,
        verbose=True,
    )

    # --- Visualization in MeshCat ---
    visualize_handle_points(meshcat, handle_points, path="handle_points")
    visualize_grasp_frames(
        meshcat,
        grasps_W,
        max_to_show=max_grasps_to_show,
        base_path="grasp_candidates",
        axis_length=0.06,
    )

    print(f"[run_handle_grasp_detection_and_visualization] "
          f"{handle_points.shape[0]} handle points, {len(grasps_W)} grasps.")

    return diagram, context, handle_points, grasps_W


meshcat.Delete()
diagram, context, handle_points, grasps_W = run_handle_grasp_detection_and_visualization(
    n_normal_neighbors=30,
    n_antipodal_candidates=3000,
    max_pt_dist=0.08,
    min_pt_dist=0.01,
    max_grasps_to_show=10,
)
# print(grasps_W)


class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context: Context, output: BasicVector) -> None:
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )
        J_G = J_G[:, self.iiwa_start : self.iiwa_end + 1]  # Only iiwa terms.
        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)

# --------------------------------------------------------------------------
# Compliant pull command: pull along drawer axis until force threshold hit
# --------------------------------------------------------------------------

class CompliantPullCommand(LeafSystem):
    """
    Outputs a spatial velocity V_WG that pulls along the drawer axis in world.
    It keeps pulling until the iiwa joint-torque norm exceeds a threshold.

    Debug: prints when switching between 'moving' and 'stopped'.
    """

    def __init__(self,
                 drawer_axis_W,
                 v_nominal: float = 0.03,   # m/s along drawer
                 tau_stop: float = 25.0):   # Nm threshold on ||tau||
        super().__init__()

        a = np.asarray(drawer_axis_W, dtype=float).reshape(3,)
        self._a = a / np.linalg.norm(a)

        self._v_nominal = v_nominal
        self._tau_stop = tau_stop

        # For debug: remember last "moving" / "stopped" mode
        self._last_moving = None

        # Inputs: iiwa joint positions (unused here but nice for future) and measured torques
        self._q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self._tau_meas_port = self.DeclareVectorInputPort("iiwa.torque_measured", 7)

        # Output: spatial velocity V_WG (6)
        self.DeclareVectorOutputPort("V_WG", 6, self.CalcOutput)

    def CalcOutput(self, context: Context, output: BasicVector) -> None:
        # We only use torques for compliance right now
        tau_meas = self._tau_meas_port.Eval(context)
        effort = float(np.linalg.norm(tau_meas))

        if effort < self._tau_stop:
            v_axis = self._v_nominal
            moving = True
        else:
            v_axis = 0.0
            moving = False

        # --- DEBUG: print when switching modes ---
        if self._last_moving is None or moving != self._last_moving:
            state_str = "MOVING" if moving else "STOPPED"
            print(
                f"[CompliantPull] t={context.get_time():.3f}  "
                f"mode={state_str}  effort={effort:.2f}  "
                f"tau_stop={self._tau_stop:.2f}  v_axis={v_axis:.3f}"
            )
            self._last_moving = moving
        # ----------------------------------------

        V_lin = v_axis * self._a
        V_ang = np.zeros(3)
        V_WG = np.hstack([V_ang, V_lin])
        output.SetFromVector(V_WG)



# --------------------------------------------------------------------------
# Switcher: trajectory phase then compliant phase
# --------------------------------------------------------------------------

class VwgSwitcher(LeafSystem):
    """
    For t < switch_time:  use trajectory V_WG
    For t >= switch_time: use compliant V_WG
    """
    def __init__(self, switch_time: float):
        super().__init__()
        self._switch_time = switch_time

        self._traj_port = self.DeclareVectorInputPort("V_WG_traj", 6)
        self._comp_port = self.DeclareVectorInputPort("V_WG_compliant", 6)
        self.DeclareVectorOutputPort("V_WG", 6, self.CalcOutput)

    def CalcOutput(self, context, output: BasicVector) -> None:
        t = context.get_time()
        if t < self._switch_time:
            V = self._traj_port.Eval(context)
        else:
            V = self._comp_port.Eval(context)
        output.SetFromVector(V)

# --------------------------------------------------------------------------
# Main setup: go to handle, close gripper, then compliant pull
# --------------------------------------------------------------------------

scenario_path = "./scenarios/bimanual_IIWA14_with_table_and_initials_and_assets_and_cameras.scenario.yaml"
scenario = LoadScenario(filename=scenario_path)
station = MakeHardwareStation(scenario, meshcat=meshcat)
builder = DiagramBuilder()
builder.AddSystem(station)

plant = station.GetSubsystemByName("plant")
temp_context = station.CreateDefaultContext()
temp_plant_context = plant.GetMyContextFromRoot(temp_context)
X_WGinitial = plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("body"))

# Hand pose that puts the camera at the "old" static camera pose
X_WG_camera = compute_wsg_camera_target_pose()

# Randomly pick a grasp from grasps_W, rotate around z by -90 degrees
drawer_grasp = grasps_W[2]
angle = np.deg2rad(-90.0)
R_z = RotationMatrix.MakeZRotation(angle)
drawer_pregrasp_rotated = drawer_grasp @ RigidTransform(R_z, [-0.1, 0.0, 0.0])
drawer_grasp_rotated = drawer_grasp @ RigidTransform(R_z, [0.05, 0.0, 0.0])

opened = 0.08
closed = -0.1

# Phase 1 keyframes:
#   1) initial home pose
#   2) move to "camera pose" (where you collected the point cloud)
#   3) move above handle
#   4) move to grasp pose (open)
#   5) close gripper
keyframes = [
    (X_WGinitial,           opened),
    (X_WG_camera,           opened),  # <-- new segment: move to camera pose
    (drawer_pregrasp_rotated, opened),
    (drawer_grasp_rotated,  opened),
    (drawer_grasp_rotated,  closed),
]

sample_times = [3 * i for i in range(len(keyframes))]  # still 3s per segment

robot_position_trajectory = PiecewisePose.MakeLinear(
    sample_times, [kf[0] for kf in keyframes]
)
traj_V_G = robot_position_trajectory.MakeDerivative()

gripper_values = np.array([kf[1] for kf in keyframes])[None]
traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, gripper_values)

V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))

# --- Compliant pull controller (Phase 2) ---

# Set this to the real drawer opening direction in world coordinates
# e.g., if the drawer opens along +x_W:
drawer_axis_W = np.array([0.0, -1.0, 0.0])

pull_cmd = builder.AddSystem(
    CompliantPullCommand(
        drawer_axis_W=drawer_axis_W,
        v_nominal=0.03,   # 3 cm/s along drawer
        tau_stop=150.0,    # Nm threshold; tune this
    )
)

# CompliantPullCommand input ports (by index)
#   0: "iiwa.position"      (size 7)
#   1: "iiwa.torque_measured" (size 7)
builder.Connect(
    station.GetOutputPort("iiwa.position_measured"),
    pull_cmd.get_input_port(0),
)
builder.Connect(
    station.GetOutputPort("iiwa.torque_measured"),
    pull_cmd.get_input_port(1),
)

# --- Switcher between trajectory and compliant commands ---

# Switch after trajectory finishes (optionally add a small buffer)
switch_time = sample_times[-1]  # or sample_times[-1] + 0.5
switcher = builder.AddSystem(VwgSwitcher(switch_time))

# VwgSwitcher input ports:
#   0: "V_WG_traj"
#   1: "V_WG_compliant"
builder.Connect(
    V_G_source.get_output_port(),        # trajectory V_WG
    switcher.get_input_port(0),
)
builder.Connect(
    pull_cmd.get_output_port(),          # compliant V_WG (single output port)
    switcher.get_input_port(1),
)

# --- Pseudo-inverse IK controller and integrator ---

controller = builder.AddSystem(PseudoInverseController(plant))
integrator = builder.AddSystem(Integrator(7))

builder.Connect(
    switcher.get_output_port(),          # selected V_WG
    controller.GetInputPort("V_WG"),
)
builder.Connect(controller.get_output_port(), integrator.get_input_port())
builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
builder.Connect(
    station.GetOutputPort("iiwa.position_measured"),
    controller.GetInputPort("iiwa.position"),
)

# WSG command
builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

# Zero extra joint torques so the Adder inside iiwa has both inputs
iiwa = plant.GetModelInstanceByName("iiwa")
nq_act = plant.num_actuated_dofs(iiwa)
zero_ff = builder.AddSystem(ConstantVectorSource(np.zeros(nq_act)))
builder.Connect(zero_ff.get_output_port(), station.GetInputPort("iiwa.torque"))

diagram = builder.Build()

# Define the simulator.
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
station_context = station.GetMyContextFromRoot(context)
integrator.set_integral_value(
    integrator.GetMyContextFromRoot(context),
    plant.GetPositions(
        plant.GetMyContextFromRoot(context),
        plant.GetModelInstanceByName("iiwa"),
    ),
)
diagram.ForcedPublish(context)
print(f"sanity check, simulation will run for {traj_V_G.end_time()} seconds")

# run simulation!
meshcat.StartRecording()
if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(traj_V_G.end_time() + 20)
meshcat.StopRecording()
meshcat.PublishRecording()
