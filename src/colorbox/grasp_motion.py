"""Simple IK-based pick-and-place routine for the gripper."""

from __future__ import annotations

import numpy as np

from pydrake.math import RigidTransform
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from .scene_setup import SimulationState
from .controllers import make_trajectory


def _solve_ik(
    sim_state: SimulationState,
    target_pose: RigidTransform,
    pos_tol: float = 0.01,
    ang_tol_rad: float = np.deg2rad(5.0),
):
    plant = sim_state.plant
    plant_context = sim_state.plant_context
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    wsg_model = plant.GetModelInstanceByName("wsg")
    wsg_frame = plant.GetFrameByName("body", wsg_model)

    ik = InverseKinematics(plant, plant_context)
    q = ik.q()
    ik.AddPositionConstraint(
        frameB=wsg_frame,
        p_BQ=np.zeros(3),
        frameA=plant.world_frame(),
        p_AQ_lower=target_pose.translation() - pos_tol,
        p_AQ_upper=target_pose.translation() + pos_tol,
    )
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=target_pose.rotation(),
        frameBbar=wsg_frame,
        R_BbarB=RigidTransform().rotation(),
        theta_bound=ang_tol_rad,
    )
    prog = ik.prog()
    q_seed = plant.GetPositions(plant_context)
    prog.SetInitialGuess(q, q_seed)
    prog.AddQuadraticErrorCost(np.eye(len(q)), q_seed, q)
    result = Solve(prog)
    if not result.is_success():
        return None
    return result.GetSolution(q)


def execute_pick_and_place(
    sim_state: SimulationState,
    grasp_pose: RigidTransform,
    drop_translation: np.ndarray | None = None,
) -> bool:
    """Move to pregrasp, grasp, move above drawer, release, return home."""
    plant = sim_state.plant
    plant_context = sim_state.plant_context
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    cabinet_model = plant.GetModelInstanceByName("cabinet")
    drawer_frame = plant.GetFrameByName("large_drawer", cabinet_model)

    wsg_model = plant.GetModelInstanceByName("wsg")
    wsg_body = plant.GetBodyByName("body", wsg_model)
    home_pose = plant.EvalBodyPoseInWorld(plant_context, wsg_body)

    pregrasp_offset = np.array([0.0, 0.0, 0.30])
    pregrasp_pose = RigidTransform(
        grasp_pose.rotation(),
        grasp_pose.translation() + pregrasp_offset,
    )

    if drop_translation is None:
        X_WD = plant.CalcRelativeTransform(
            plant_context, plant.world_frame(), drawer_frame
        )
        drop_pose = X_WD @ RigidTransform([0.0, -0.35, -0.1])
    else:
        drop_pose = RigidTransform(
            grasp_pose.rotation(),
            drop_translation,
        )

    opened = 0.107
    closed = 0.0

    key_frames = [
        ("initial", home_pose, opened),
        ("pregrasp", pregrasp_pose, opened),
        ("grasp", grasp_pose, opened),
        ("grasping", grasp_pose, closed),
        ("after_grasp", pregrasp_pose, closed),
        ("lift", drop_pose, closed),
        ("release", drop_pose, opened),
        ("return_home", home_pose, opened),
    ]

    gripper_poses = [kf[1] for kf in key_frames]
    finger_states = np.asarray([kf[2] for kf in key_frames]).reshape(1, -1)
    sample_times = [i * 3.0 for i in range(len(gripper_poses))]

    traj_velocity, traj_wsg_command = make_trajectory(gripper_poses, finger_states, sample_times)

    if sim_state.velocity_source is None:
        # Fall back to IK stepping if velocity control is not enabled.
        for name, pose, fingers in key_frames:
            q_sol = _solve_ik(sim_state, pose)
            if q_sol is None:
                print(f"IK failed for {name} pose; aborting pick-and-place.")
                return False
            sim_state.set_arm_configuration(q_sol)
            sim_state.set_gripper_opening(fingers)
            sim_state.advance(2.0)
        return True

    # Stream trajectories through the pseudoinverse controller.
    dt = 0.05
    t = traj_velocity.start_time()
    t_end = traj_velocity.end_time()
    while t <= t_end + 1e-6:
        V_cmd = traj_velocity.value(t).ravel()
        wsg_cmd = traj_wsg_command.value(t)[0]
        sim_state.set_gripper_twist(V_cmd)
        sim_state.set_gripper_opening(wsg_cmd)
        sim_state.advance(dt)
        t += dt
    # Zero twist at end.
    sim_state.set_gripper_twist([0, 0, 0, 0, 0, 0])
    return True
