"""Executable entry point that mirrors the notebook workflow."""

from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(__file__))

import numpy as np

from camera_system import CameraSystem
from grasp_motion import execute_pick_and_place
from grasp_planning import find_best_antipodal_grasp
from perception import (
    bayesian_task_clouds,
    collect_multiview_data,
    cosine_average_task_clouds,
    load_perception_assets,
    simple_sam_clip_pipeline,
)
from scene_setup import (
    build_simulation,
    build_station_setup,
    initialize_drawer_boxes,
)

from pydrake.math import RigidTransform, RotationMatrix

device = "cpu"  # Change to "cpu" if no GPU is available.


def main() -> None:
    setup = build_station_setup(use_velocity_control=True)
    sim_state = build_simulation(setup)
    initialize_drawer_boxes(sim_state)

    # Warm start the simulation.
    sim_state.advance(2.0)

    camera = CameraSystem(0, sim_state.diagram, sim_state.diagram_context)
    assets = load_perception_assets(device=device)

    dt = 0.5
    q_checkpoints = np.array(
        [
            [-3 * np.pi / 8.0, np.pi / 4.0, 0.0, -2 * np.pi / 6.0, 0.0, 3 * np.pi / 6.0, np.pi / 2.0],
            [
                -3 * np.pi / 8.0 + 0.5,
                np.pi / 4.0,
                0.0,
                -2 * np.pi / 6.0,
                -0.5,
                3 * np.pi / 6.0,
                np.pi / 2.0,
            ],
            [
                -3 * np.pi / 8.0 + 1.0,
                np.pi / 4.0,
                0.0,
                -2 * np.pi / 6.0,
                -0.7,
                3 * np.pi / 6.0 + 0.2,
                np.pi / 2.0,
            ],
        ]
    )

    # Perception pipeline (uses teleporting joint targets even under velocity control).
    # simple_clusters, _, concat_pcd = simple_sam_clip_pipeline(
    #     sim_state, camera, q_checkpoints, assets, dt=dt
    # )

    multiview_data = collect_multiview_data(
        sim_state, camera, q_checkpoints, assets, dt=dt
    )
    cosine_clusters, concat_pcd_mv, _ = cosine_average_task_clouds(
        multiview_data, camera, assets.tasks
    )
    bayes_clusters, _, probs = bayesian_task_clouds(
        multiview_data, camera, assets
    )

    grasp_pose = find_best_antipodal_grasp(bayes_clusters, concat_pcd_mv, sim_state.meshcat)
    if grasp_pose is None:
        print("No grasp pose found. Check segmentation results and thresholds.")
        return

    print("Found feasible grasp pose:", grasp_pose)
    execute_pick_and_place(sim_state, grasp_pose)
    print("Pick and place execution completed.")


if __name__ == "__main__":
    main()
