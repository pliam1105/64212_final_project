"""Drawer opening execution: grasp detection, planning, and compliant pull."""

from __future__ import annotations

import numpy as np
from pydrake.all import (
    PiecewisePolynomial,
    PiecewisePose,
    TrajectorySource,
    Simulator,
    Integrator,
    ConstantVectorSource,
    PointCloud,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder

from open_drawer_RANSAC import run_handle_grasp_detection_and_visualization
from controller import PseudoInverseController, CompliantPullCommand, VwgSwitcher


def open_drawer(
    builder: DiagramBuilder,
    station: object,
    meshcat: object,
    pc: PointCloud,
) -> str:
    """Execute the drawer-opening sequence.
    
    Args:
        builder: DiagramBuilder (will be extended with trajectory/controller systems)
        station: The hardware station system (already added to builder)
        meshcat: Meshcat instance for visualization and recording
        pc: PointCloud from preRANSAC (captured from camera0_point_cloud)
    
    Returns:
        New state name ("done")
    """
    if pc is None:
        raise ValueError("Point cloud 'pc' must be provided to open_drawer.")

    # For backwards compatibility, call the planning helper and then
    # build/run the execution graph as before.
    traj_V_G, traj_wsg_command, switch_time, pull_params = plan_open_drawer(
        station=station, pc=pc, meshcat=meshcat
    )

    # Build execution phase using the provided builder (same as before)
    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))

    # Add compliant pull controller
    pull_cmd = builder.AddSystem(
        CompliantPullCommand(**pull_params)
    )

    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        pull_cmd.get_input_port(0),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.torque_measured"),
        pull_cmd.get_input_port(1),
    )

    # Switcher (trajectory -> compliant pull)
    switcher = builder.AddSystem(VwgSwitcher(switch_time))
    builder.Connect(V_G_source.get_output_port(), switcher.get_input_port(0))
    builder.Connect(pull_cmd.get_output_port(), switcher.get_input_port(1))

    # Pseudo-inverse IK controller and integrator
    plant = station.GetSubsystemByName("plant")
    controller = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(7))

    builder.Connect(switcher.get_output_port(), controller.GetInputPort("V_WG"))
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        controller.GetInputPort("iiwa.position"),
    )

    # Gripper command
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

    # Note: Station is in position-only control mode; no torque input port exists.

    # Build and run as before
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    integrator_sys = diagram.GetSystemByName("Integrator")
    integrator_sys.set_integral_value(
        integrator_sys.GetMyContextFromRoot(context),
        plant.GetPositions(
            plant.GetMyContextFromRoot(context),
            plant.GetModelInstanceByName("iiwa"),
        ),
    )

    diagram.ForcedPublish(context)
    print(f"[open_drawer] Simulation will run for {traj_V_G.end_time()} seconds")

    simulator = Simulator(diagram, context)
    meshcat.StartRecording()
    simulator.AdvanceTo(traj_V_G.end_time() + 20)
    meshcat.StopRecording()
    meshcat.PublishRecording()

    return "done"


def plan_open_drawer(
    *,
    station: object,
    pc: PointCloud,
    meshcat: object,
    diagram_context: object = None,
    n_normal_neighbors: int = 30,
    n_antipodal_candidates: int = 3000,
    max_pt_dist: float = 0.08,
    min_pt_dist: float = 0.01,
    max_grasps_to_show: int = 10,
) -> tuple:
    """Plan trajectories for opening the drawer using the provided point cloud.

    Args:
        station: Hardware station system
        pc: PointCloud from perception
        meshcat: Meshcat for visualization
        diagram_context: Context from the persistent diagram (required when station is part of a diagram)
        n_normal_neighbors: Number of neighbors for normal estimation
        n_antipodal_candidates: Number of grasp candidates
        max_pt_dist, min_pt_dist: Distance thresholds
        max_grasps_to_show: Max grasps to visualize

    Returns a tuple (traj_V_G, traj_wsg_command, switch_time, pull_params).
    This function performs perception and computes the task-space and gripper
    trajectories but does not modify or build execution diagrams.
    """
    # Run perception / grasp detection
    handle_points, grasps_W = run_handle_grasp_detection_and_visualization(
        n_normal_neighbors=n_normal_neighbors,
        n_antipodal_candidates=n_antipodal_candidates,
        max_pt_dist=max_pt_dist,
        min_pt_dist=min_pt_dist,
        max_grasps_to_show=max_grasps_to_show,
        pc=pc,
        meshcat=meshcat,
    )

    # Current robot pose
    plant = station.GetSubsystemByName("plant")
    if diagram_context is None:
        # Standalone mode: create a context for station alone
        temp_context = station.CreateDefaultContext()
        temp_plant_context = plant.GetMyContextFromRoot(temp_context)
    else:
        # Persistent diagram mode: extract plant context from the diagram context
        temp_plant_context = plant.GetMyContextFromRoot(diagram_context)
    X_WGinitial = plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("body"))

    # Choose a grasp and create keyframes
    drawer_grasp = grasps_W[-1]
    angle = np.deg2rad(-90.0)
    R_z = RotationMatrix.MakeZRotation(angle)
    drawer_pregrasp_rotated = drawer_grasp @ RigidTransform(R_z, [-0.1, 0.0, 0.0])
    drawer_grasp_rotated = drawer_grasp @ RigidTransform(R_z, [0.05, 0.0, 0.0])

    opened = 0.08
    closed = -0.1

    keyframes = [
        (X_WGinitial, opened),
        (drawer_pregrasp_rotated, opened),
        (drawer_grasp_rotated, opened),
        (drawer_grasp_rotated, closed),
    ]

    sample_times = [3 * i for i in range(len(keyframes))]

    robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, [kf[0] for kf in keyframes])
    traj_V_G = robot_position_trajectory.MakeDerivative()

    gripper_values = np.array([kf[1] for kf in keyframes])[None]
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, gripper_values)

    switch_time = sample_times[-1]

    pull_params = dict(
        drawer_axis_W=np.array([0.0, -1.0, 0.0]),
        v_nominal=0.03,
        tau_stop=150.0,
    )

    return traj_V_G, traj_wsg_command, switch_time, pull_params


