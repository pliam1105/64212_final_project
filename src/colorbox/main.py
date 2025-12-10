"""Executable entry point that mirrors the notebook workflow."""

from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(__file__))

import numpy as np

from camera_system import CameraSystem
from grasp_motion import execute_pick_and_place
from grasp_planning import find_best_antipodal_grasp, evaluate_box_grasp
from perception import (
    bayesian_task_clouds,
    collect_multiview_data,
    cosine_average_task_clouds,
    load_perception_assets,
    simple_sam_clip_pipeline,
    sam3_pipeline,
    load_sam3,
    evaluate_box_segmentation,
)
from scene_setup import (
    build_simulation,
    build_station_setup,
    initialize_drawer_boxes,
    initialize_random_drawer_boxes,
)

from pydrake.math import RigidTransform, RotationMatrix

device = "cuda:0"  # Change to "cpu" if no GPU is available.

NUM_SCENARIOS = 20

def main() -> None:
    simple_eval = []
    cosine_eval = []
    bayes_eval = []
    sam3_eval = []
    grasp_simple_eval = []
    grasp_cosine_eval = []
    grasp_bayes_eval = []
    grasp_sam3_eval = []
    for scenario_num in range(NUM_SCENARIOS):
        setup = build_station_setup(use_velocity_control=True)
        sim_state = build_simulation(setup)
        # initialize_drawer_boxes(sim_state)
        initialize_random_drawer_boxes(sim_state)

        # Warm start the simulation.
        sim_state.advance(2.0)

        camera = CameraSystem(0, sim_state.diagram, sim_state.diagram_context)
        samclip_assets = load_perception_assets(device=device)
        sam3_assets = load_sam3(device=device)

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
        simple_clusters, _, concat_pcd, simple_task_ids = simple_sam_clip_pipeline(
            sim_state, camera, q_checkpoints, samclip_assets, dt=dt
        )

        multiview_data = collect_multiview_data(
            sim_state, camera, q_checkpoints, samclip_assets, dt=dt
        )
        cosine_clusters, concat_pcd_mv, _, cosine_task_ids = cosine_average_task_clouds(
            multiview_data, camera, samclip_assets.tasks
        )
        bayes_clusters, concat_pcd_mv, probs, bayes_task_ids = bayesian_task_clouds(
            multiview_data, camera, samclip_assets
        )

        sam3_clusters, _, _, sam3_task_ids = sam3_pipeline(
            sim_state, camera, q_checkpoints, sam3_assets, dt=dt
        )

        simple_task_cluster_nums = evaluate_box_segmentation(sim_state.plant, sim_state.plant_context, simple_clusters, simple_task_ids)
        print(f'simple task cluster nums: {simple_task_cluster_nums}')
        
        cosine_task_cluster_nums = evaluate_box_segmentation(sim_state.plant, sim_state.plant_context, cosine_clusters, cosine_task_ids)
        print(f'cosine task cluster nums: {cosine_task_cluster_nums}')

        bayes_task_cluster_nums = evaluate_box_segmentation(sim_state.plant, sim_state.plant_context, bayes_clusters, bayes_task_ids)
        print(f'bayes task cluster nums: {bayes_task_cluster_nums}')

        sam3_task_cluster_nums = evaluate_box_segmentation(sim_state.plant, sim_state.plant_context, sam3_clusters, sam3_task_ids)
        print(f'sam3 task cluster nums: {sam3_task_cluster_nums}')

        simple_eval.append(simple_task_cluster_nums)
        cosine_eval.append(cosine_task_cluster_nums)
        bayes_eval.append(bayes_task_cluster_nums)
        sam3_eval.append(sam3_task_cluster_nums)

        simple_grasp_nums = evaluate_box_grasp(simple_clusters, simple_task_ids, concat_pcd, sim_state.meshcat)
        print(f'simple grasp nums: {simple_grasp_nums}')
        
        cosine_grasp_nums = evaluate_box_grasp(cosine_clusters, cosine_task_ids, concat_pcd, sim_state.meshcat)
        print(f'cosine grasp nums: {cosine_grasp_nums}')

        bayes_grasp_nums = evaluate_box_grasp(bayes_clusters, bayes_task_ids, concat_pcd, sim_state.meshcat)
        print(f'bayes grasp nums: {bayes_grasp_nums}')

        sam3_grasp_nums = evaluate_box_grasp(sam3_clusters, sam3_task_ids, concat_pcd, sim_state.meshcat)
        print(f'sam3 grasp nums: {sam3_grasp_nums}')

        grasp_simple_eval.append(simple_grasp_nums)
        grasp_cosine_eval.append(cosine_grasp_nums)
        grasp_bayes_eval.append(bayes_grasp_nums)
        grasp_sam3_eval.append(sam3_grasp_nums)

    np.savez('perception_eval.npz', **{'simple': simple_eval, 'cosine': cosine_eval, 'bayes': bayes_eval, 'sam3': sam3_eval})
    np.savez('grasp_eval.npz', **{'grasp_simple': grasp_simple_eval, 'grasp_cosine': grasp_cosine_eval, 'grasp_bayes': grasp_bayes_eval, 'grasp_sam3': grasp_sam3_eval})

    # grasp_pose = find_best_antipodal_grasp(bayes_clusters, concat_pcd_mv, sim_state.meshcat)
    # grasp_pose = find_best_antipodal_grasp(simple_clusters, concat_pcd, sim_state.meshcat)
    # grasp_pose = find_best_antipodal_grasp(sam3_clusters, concat_pcd, sim_state.meshcat)
    # if grasp_pose is None:
    #     print("No grasp pose found. Check segmentation results and thresholds.")
    #     return

    # print("Found feasible grasp pose:", grasp_pose)
    # execute_pick_and_place(sim_state, grasp_pose)
    # print("Pick and place execution completed.")


if __name__ == "__main__":
    main()
