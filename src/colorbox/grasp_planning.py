"""Grasp planning helpers converted from the original notebook."""

from __future__ import annotations

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
)

from manipulation.clutter import GenerateAntipodalGraspCandidate
from manipulation.station import ConfigureParser


def find_best_antipodal_grasp(
    task_clusters,
    concat_pcd,
    meshcat,
    num_iterations: int = 100,
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.dmd.yaml")
    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.prefix = "planning"
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    plant_context = plant.GetMyContextFromRoot(context)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    gripper_body = plant.GetBodyByName("body")
    margin = 0.0
    min_total_cost = np.inf
    best_grasp_pose = None

    for task_cluster in task_clusters:
        if task_cluster.size() == 0:
            continue
        rng = np.random.default_rng()
        min_cost = np.inf
        task_grasp_pose = None
        for _ in range(num_iterations):
            cost, pose = GenerateAntipodalGraspCandidate(
                diagram, context, task_cluster, rng
            )
            if pose is None or not np.isfinite(cost):
                continue
            plant.SetFreeBodyPose(plant_context, gripper_body, pose)
            query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
            for i in range(concat_pcd.size()):
                distances = query_object.ComputeSignedDistanceToPoint(
                    concat_pcd.xyz(i), threshold=margin
                )
                if distances:
                    cost = np.inf
                    break
            if np.isfinite(cost) and cost < min_cost:
                min_cost = cost
                task_grasp_pose = pose
        if np.isfinite(min_cost) and min_cost < min_total_cost:
            min_total_cost = min_cost
            best_grasp_pose = task_grasp_pose

    if best_grasp_pose is not None:
        plant.SetFreeBodyPose(
            plant_context, plant.GetBodyByName("body"), best_grasp_pose
        )
        diagram.ForcedPublish(context)
    return best_grasp_pose
