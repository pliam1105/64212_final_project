from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from ultralytics import FastSAM

import clip
from pydrake.all import BaseField, Concatenate, Fields, PointCloud, PiecewisePose

from camera_system import CameraSystem

import os

@dataclass
class PerceptionAssets:
    mask_generator: FastSAM
    clip_model: torch.nn.Module
    preprocess: any
    tasks: np.ndarray
    task_embeddings: np.ndarray
    normalized_task_embeddings: np.ndarray
    task_embeddings_torch: torch.Tensor
    device: str

def load_perception_assets(
    task_list_path: str | Path = "task_list.txt",
    fastsam_weights: str = "FastSAM-x.pt",
    clip_model_name: str = "ViT-L/14",
    device: str = "cuda",
) -> PerceptionAssets:
    tasks = np.genfromtxt(task_list_path, delimiter="\n", dtype=str)
    mask_generator = FastSAM(fastsam_weights)
    mask_generator.eval()
    clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model.eval()
    with torch.no_grad():
        task_embeddings = (
            clip_model.encode_text(clip.tokenize(tasks).to(device)).cpu().numpy()
        )
    normalized_task_embeddings = task_embeddings / np.linalg.norm(
        task_embeddings, axis=1, keepdims=True
    )
    task_embeddings_torch = torch.tensor(task_embeddings, device=device)
    return PerceptionAssets(
        mask_generator=mask_generator,
        clip_model=clip_model,
        preprocess=preprocess,
        tasks=tasks,
        task_embeddings=task_embeddings,
        normalized_task_embeddings=normalized_task_embeddings,
        task_embeddings_torch=task_embeddings_torch,
        device=device,
    )

def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    min_x = int(np.min(np.where(mask == 1)[0]))
    min_y = int(np.min(np.where(mask == 1)[1]))
    max_x = int(np.max(np.where(mask == 1)[0]))
    max_y = int(np.max(np.where(mask == 1)[1]))
    return min_x, min_y, max_x, max_y


def cropped_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    min_x, min_y, max_x, max_y = bbox_from_mask(mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    return masked_image[min_x:max_x, min_y:max_y]


def get_clip_embedding(image: np.ndarray, assets: PerceptionAssets) -> torch.Tensor:
    with torch.no_grad():
        preproc_img = assets.preprocess(Image.fromarray(image)).unsqueeze(0).to(
            assets.device
        )
        return assets.clip_model.encode_image(preproc_img)


def cosine_similarity_to_tasks(
    img_emb: np.ndarray, normalized_task_embeddings: np.ndarray
) -> np.ndarray:
    emb = img_emb.reshape(-1)
    emb_norm = np.linalg.norm(emb) + 1e-8
    return normalized_task_embeddings @ (emb / emb_norm)


def _voxel_downsample(pcd: PointCloud, voxel_size: float) -> PointCloud:
    return pcd.VoxelizedDownSample(voxel_size=voxel_size)


def _new_cloud(num_pts: int) -> PointCloud:
    return PointCloud(num_pts, Fields(BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals))


def _cluster_cloud(points: PointCloud, eps: float, min_points: int) -> PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.xyzs().T)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_cluster_num = -1
    max_cluster = None
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) > max_cluster_num:
            max_cluster_num = len(indices)
            max_cluster = pcd.select_by_index(indices.tolist())
    if max_cluster_num == -1:
        return PointCloud(0, Fields(BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals))
    max_cluster.estimate_normals()
    clustered = PointCloud(
        len(max_cluster.points), Fields(BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals)
    )
    clustered.mutable_xyzs()[:] = np.asarray(max_cluster.points).T
    normals = np.asarray(max_cluster.normals)
    if normals.shape[0] == len(max_cluster.points):
        clustered.mutable_normals()[:] = normals.T
    return clustered

@dataclass
class MultiviewData:
    cam_poses: list[np.ndarray]
    segments: list[list[np.ndarray]]
    cos_sims: list[list[np.ndarray]]
    depth_ims: list[np.ndarray]
    concat_pcd: PointCloud

def cosine_average_task_clouds(
    data: MultiviewData,
    camera: CameraSystem,
    tasks: Sequence[str],
    cosine_threshold: float = 0.2,
    cluster_dist: float = 0.01,
    occlusion_threshold: float = 0.01,
) -> tuple[list[PointCloud], PointCloud, np.ndarray]:
    concat_pcd = data.concat_pcd
    cos_s = np.zeros((concat_pcd.size(), len(tasks)))
    num_emb = np.zeros((concat_pcd.size(),))

    for camera_pose, segments_img, cos_sims_img, depth_im in zip(
        data.cam_poses, data.segments, data.cos_sims, data.depth_ims
    ):
        homogeneous_pcl = np.vstack([concat_pcd.xyzs(), np.ones((concat_pcd.size()))])
        cam_frame_h = (np.linalg.inv(camera_pose) @ homogeneous_pcl).T
        w = cam_frame_h[:, 3]
        w_safe = np.where(np.abs(w) < 1e-9, np.nan, w)
        pcl_in_cam_frame = cam_frame_h[:, :3] / w_safe[:, None]
        projected_pcl = camera.project_pC_to_img(pcl_in_cam_frame)
        for segment, cos_sim in zip(segments_img, cos_sims_img):
            if np.max(cos_sim) <= cosine_threshold:
                continue
            in_bounds = (
                (projected_pcl[:, 0] >= 0)
                & (projected_pcl[:, 0] < segment.shape[1])
                & (projected_pcl[:, 1] >= 0)
                & (projected_pcl[:, 1] < segment.shape[0])
            )
            not_occluded = np.full_like(in_bounds, False)
            idx = np.where(in_bounds)[0]
            not_occluded[idx] = (
                np.abs(
                    pcl_in_cam_frame[idx, 2]
                    - depth_im[
                        (projected_pcl[idx, 1], projected_pcl[idx, 0])
                    ]
                )
                < occlusion_threshold
            )
            pcl_mask = np.full_like(in_bounds, False)
            pcl_mask[idx] = segment[(projected_pcl[idx, 1], projected_pcl[idx, 0])] == 1
            pcl_mask &= not_occluded
            indices = np.where(pcl_mask)[0]
            if len(indices) == 0:
                continue
            cos_s[indices] = (
                cos_s[indices] * num_emb[indices].reshape(len(indices), 1)
                + cos_sim.reshape(1, len(tasks))
            ) / (num_emb[indices].reshape(len(indices), 1) + 1)
            num_emb[indices] += 1

    task_ids = np.argmax(cos_s, axis=1)
    max_prob = np.max(cos_s, axis=1)
    task_ids = np.where(max_prob > cosine_threshold, task_ids, -1)

    combined_task_pcls = [_new_cloud(0) for _ in tasks]
    for task_id in range(len(tasks)):
        indices = np.where(task_ids == task_id)[0]
        task_points = concat_pcd.xyzs().T[indices, :]
        combined_task_pcls[task_id].resize(len(task_points))
        if len(task_points) > 0:
            combined_task_pcls[task_id].mutable_xyzs()[:] = task_points.T
            combined_task_pcls[task_id].mutable_rgbs()[0] = 255 * task_id / len(tasks)
            combined_task_pcls[task_id].mutable_rgbs()[2] = 255 * (1 - task_id / len(tasks))

    task_clusters = []
    for task_id in range(len(tasks)):
        clustered = _cluster_cloud(
            combined_task_pcls[task_id], eps=cluster_dist, min_points=10
        )
        if clustered.size() > 0 and combined_task_pcls[task_id].size() > 0:
            clustered.mutable_rgbs()[0] = combined_task_pcls[task_id].rgbs()[0, 0]
            clustered.mutable_rgbs()[1] = combined_task_pcls[task_id].rgbs()[1, 0]
            clustered.mutable_rgbs()[2] = combined_task_pcls[task_id].rgbs()[2, 0]
        task_clusters.append(clustered)

    return task_clusters, concat_pcd, cos_s


torch_pi = torch.tensor(np.pi)
YES_MEAN = 0.27
YES_STD_DEV = 0.035
NO_MEAN = 0.20
NO_STD_DEV = 0.035
PRIOR = 0.5

NUM_TASKS = len(np.genfromtxt(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "task_list.txt"), delimiter="\n", dtype=str))


def fast_density(x, std, mean):
    return (1 / (std * torch.sqrt(2 * torch_pi.to(x.device)))) * torch.exp(
        -0.5 * ((x - mean) / std) ** 2
    )


def bayesian_update(posterior_y1_batch, cosine_batch, alpha: float = 0.1):
    density_yes = 0.5
    density_no = 1 - density_yes
    p_x_yes = density_yes * fast_density(cosine_batch, YES_STD_DEV, YES_MEAN)
    p_x_no = density_no * fast_density(cosine_batch, NO_STD_DEV, NO_MEAN)
    ratio = p_x_yes / p_x_no
    posterior_y0 = 1 - posterior_y1_batch
    y1 = posterior_y1_batch * p_x_yes / (posterior_y1_batch * p_x_yes + posterior_y0 * p_x_no)
    mask = (ratio < alpha).all()
    y1[:, mask] = float("nan")
    return torch.where(torch.isnan(y1), posterior_y1_batch, y1)


def bayesian_task_clouds(
    data: MultiviewData,
    camera: CameraSystem,
    assets: PerceptionAssets,
    cluster_dist: float = 0.025,
    occlusion_threshold: float = 0.01,
    prob_threshold: float = 1.0/NUM_TASKS+0.1,
) -> tuple[list[PointCloud], PointCloud, np.ndarray]:
    concat_pcd = data.concat_pcd
    probs = torch.full((concat_pcd.size(), len(assets.tasks)), PRIOR, device=assets.device)

    for camera_pose, segments_img, cos_sims_img, depth_im in zip(
        data.cam_poses, data.segments, data.cos_sims, data.depth_ims
    ):
        homogeneous_pcl = np.vstack([concat_pcd.xyzs(), np.ones((concat_pcd.size()))])
        cam_frame_h = (np.linalg.inv(camera_pose) @ homogeneous_pcl).T
        w = cam_frame_h[:, 3]
        w_safe = np.where(np.abs(w) < 1e-9, np.nan, w)
        pcl_in_cam_frame = cam_frame_h[:, :3] / w_safe[:, None]
        projected_pcl = camera.project_pC_to_img(pcl_in_cam_frame)
        for segment, cos_sim in zip(segments_img, cos_sims_img):
            in_bounds = (
                (projected_pcl[:, 0] >= 0)
                & (projected_pcl[:, 0] < segment.shape[1])
                & (projected_pcl[:, 1] >= 0)
                & (projected_pcl[:, 1] < segment.shape[0])
            )
            not_occluded = np.full_like(in_bounds, False)
            idx = np.where(in_bounds)[0]
            not_occluded[idx] = (
                np.abs(
                    pcl_in_cam_frame[idx, 2]
                    - depth_im[
                        (projected_pcl[idx, 1], projected_pcl[idx, 0])
                    ]
                )
                < occlusion_threshold
            )
            pcl_mask = np.full_like(in_bounds, False)
            pcl_mask[idx] = segment[(projected_pcl[idx, 1], projected_pcl[idx, 0])] == 1
            pcl_mask &= not_occluded
            indices = np.where(pcl_mask)[0]
            if len(indices) == 0:
                continue
            probs[indices] = bayesian_update(
                probs[indices],
                torch.tensor(cos_sim.reshape(1, len(assets.tasks)), device=assets.device),
            )

    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    task_ids = torch.argmax(probs, dim=1).cpu().numpy()
    max_prob = torch.max(probs, dim=1).values.cpu().numpy()
    task_ids = np.where(max_prob > prob_threshold, task_ids, -1)
    prob_avg = [0.0 for _ in range(len(assets.tasks))]

    combined_task_pcls = [_new_cloud(0) for _ in assets.tasks]
    for task_id in range(len(assets.tasks)):
        indices = np.where(task_ids == task_id)[0]
        task_points = concat_pcd.xyzs().T[indices, :]
        combined_task_pcls[task_id].resize(len(task_points))
        if len(task_points) > 0:
            combined_task_pcls[task_id].mutable_xyzs()[:] = task_points.T
            combined_task_pcls[task_id].mutable_rgbs()[0] = 255 * task_id / len(assets.tasks)
            combined_task_pcls[task_id].mutable_rgbs()[2] = 255 * (1 - task_id / len(assets.tasks))
        prob_avg[task_id] = np.mean(max_prob[indices])
        print(f'# pts in task {task_id} -> {combined_task_pcls[task_id].size()}, avg prob: {prob_avg[task_id]}')

    task_clusters = []
    for task_id in range(len(assets.tasks)):
        clustered = _cluster_cloud(
            combined_task_pcls[task_id], eps=cluster_dist, min_points=3
        )
        if clustered.size() > 0 and combined_task_pcls[task_id].size() > 0:
            clustered.mutable_rgbs()[0] = combined_task_pcls[task_id].rgbs()[0, 0]
            clustered.mutable_rgbs()[1] = combined_task_pcls[task_id].rgbs()[1, 0]
            clustered.mutable_rgbs()[2] = combined_task_pcls[task_id].rgbs()[2, 0]
        task_clusters.append(clustered)

    return task_clusters, concat_pcd, probs.cpu().numpy()


def collect_multiview_data(
    camera: CameraSystem,
    q_checkpoints: np.ndarray,
    assets: PerceptionAssets,
    dt: float = 0.5,
    voxel_size: float = 0.005,
    integrator: Optional[object] = None,
    context: Optional[object] = None,
    diagram: Optional[object] = None,
    simulator: Optional[object] = None,
    station_system: object = None,
    V_G_source: object = None,
    wsg_source: object = None,
) -> MultiviewData:
    
    mask_generator = assets.mask_generator
    normalized_tasks = assets.normalized_task_embeddings

    cam_poses: list[np.ndarray] = []
    segments: list[list[np.ndarray]] = []
    cos_sims: list[list[np.ndarray]] = []
    depth_ims: list[np.ndarray] = []
    concat_pcd = _new_cloud(0)

    plant = station_system.GetSubsystemByName("plant")
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    wsg_body = plant.GetBodyByName("body")
    
    # Ensure WSG is open for the entire multiview capture phase
    if wsg_source is not None:
        from pydrake.all import PiecewisePolynomial
        # Create a constant trajectory that keeps gripper fully open (0.1 m)
        wsg_open_vals = np.full((1, 2), 0.08)
        wsg_open_traj = PiecewisePolynomial.FirstOrderHold([0.0, 100.0], wsg_open_vals)
        wsg_source.UpdateTrajectory(wsg_open_traj)
        print("[collect_multiview_data] WSG set to open (0.1 m) for entire phase")

    for i, q_target in enumerate(q_checkpoints):
        print(f"[_enter_SAM] Moving to checkpoint {i+1}/{len(q_checkpoints)}")

        plant_ctx = diagram.GetMutableSubsystemContext(plant, context)
        q_current = plant.GetPositions(plant_ctx, iiwa_model)
        print(f"  Current q: {q_current}")
        print(f"  Target q:  {q_target}")
        
        # Create a smooth trajectory from current to target (3 second motion)
        # Times are relative to the START of this trajectory phase
        sample_times = [0.0, 3.0]
        wsg_body = plant.GetBodyByName("body")
        X_current = plant.EvalBodyPoseInWorld(plant_ctx, wsg_body)
        
        # Compute target EE pose via FK
        plant.SetPositions(plant_ctx, iiwa_model, q_target)
        X_target = plant.EvalBodyPoseInWorld(plant_ctx, wsg_body)
        
        # Create trajectory
        robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, [X_current, X_target])
        traj_V_G = robot_position_trajectory.MakeDerivative()
        
        # Update persistent trajectory source
        V_G_source.UpdateTrajectory(traj_V_G)
        print(f"  Updated V_G trajectory, end_time={traj_V_G.end_time()}")

        # Initialize integrator to current state
        integ_ctx = integrator.GetMyContextFromRoot(context)
        integrator.set_integral_value(integ_ctx, q_current)
        print(f"  Integrator initialized to q_current")
        
        # CRITICAL: Sync plant context with integrator state before advancing
        # This ensures the plant starts from the correct position
        plant_ctx = diagram.GetMutableSubsystemContext(plant, context)
        plant.SetPositions(plant_ctx, iiwa_model, q_current)
        
        # Publish to update visualization
        diagram.ForcedPublish(context)
        print(f"  Published at t={context.get_time()}")
        
        # Advance simulator to execute trajectory
        start_time = context.get_time()
        target_time = start_time + 5  # Match trajectory duration
        print(f"  Advancing simulator from t={start_time:.2f} to t={target_time:.2f}")
        simulator.AdvanceTo(target_time)
        print(f"  Simulation advanced, now at t={context.get_time():.2f}")
        
        # Dwell for sensor capture (no motion, just camera snapshot)
        dwell_time = context.get_time() + 0.5
        simulator.AdvanceTo(dwell_time)
        print(f"  Dwelled until t={context.get_time():.2f}")

        camera.update_camera_feed()
        camera.update_camera_pos()

        rgb_im = camera.rgb_im[:, :, :3]
        depth_im = camera.depth_im
        camera_pose = camera.X_WC.GetAsMatrix4().copy()

        cam_poses.append(camera_pose)
        depth_ims.append(depth_im)

        x_coords, y_coords = np.meshgrid(
            np.arange(rgb_im.shape[0]),
            np.arange(rgb_im.shape[1]),
            indexing="ij",
        )
        x_coords = x_coords.reshape(-1)
        y_coords = y_coords.reshape(-1)
        depths = np.vstack([x_coords, y_coords, depth_im.reshape(-1)]).T
        depths = depths[np.where(depths[:, 2] != 10.0)[0]]
        pcl = camera.project_depth_to_pC(depths)
        homogeneous_pcl = np.vstack([pcl.T, np.ones((len(pcl)))])
        global_pcl = (camera_pose @ homogeneous_pcl).T
        global_pcl = global_pcl[:, :3] / global_pcl[:, 3:]

        pcd = _new_cloud(len(global_pcl))
        pcd.mutable_xyzs()[:] = global_pcl.T
        pcd.mutable_rgbs()[:] = np.array([[0, 0, 255]] * len(global_pcl)).T
        pcd = _voxel_downsample(pcd, voxel_size)
        concat_pcd = Concatenate([concat_pcd, pcd])

        sam_results = mask_generator(rgb_im)
        masks = sam_results[0].masks
        segments_img = []
        cos_sims_img = []
        for mask in masks:
            mask_2d = mask.data.cpu().numpy()[0, :, :]
            mask_2d = cv2.resize(mask_2d, (rgb_im.shape[1], rgb_im.shape[0]))
            segments_img.append(mask_2d)
            cropped_img = cropped_mask(rgb_im, mask_2d)
            img_emb = get_clip_embedding(cropped_img, assets).cpu().numpy()
            cos_scores = cosine_similarity_to_tasks(img_emb, normalized_tasks)
            cos_sims_img.append(cos_scores)
        segments.append(segments_img)
        cos_sims.append(cos_sims_img)

    concat_pcd = _voxel_downsample(concat_pcd, voxel_size)
    return MultiviewData(
        cam_poses=cam_poses,
        segments=segments,
        cos_sims=cos_sims,
        depth_ims=depth_ims,
        concat_pcd=concat_pcd,
    )