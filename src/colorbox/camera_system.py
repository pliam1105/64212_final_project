"""Camera helper that exposes RGB-D data and projection utilities."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


class CameraSystem:
    """Lightweight wrapper around the camera subsystem exported by Drake."""

    def __init__(self, idx: int, diagram, context):
        self.idx = idx
        self.diagram = diagram
        self.context = context
        self.station = self.diagram.GetSubsystemByName("station")
        self.station_context = self.station.GetMyMutableContextFromRoot(self.context)
        self.cam = self.station.GetSubsystemByName(f"rgbd_sensor_camera{idx}")
        self.cam_context = self.cam.GetMyMutableContextFromRoot(self.context)
        self.X_WC = self.cam.body_pose_in_world_output_port().Eval(self.cam_context)
        self.cam_info = self.cam.default_depth_render_camera().core().intrinsics()

        self.depth_im = None
        self.rgb_im = None
        self.update_camera_feed()

    def project_depth_to_pC(self, depth_pixel: np.ndarray) -> np.ndarray:
        """Project image pixels + depth into the camera frame."""

        v = depth_pixel[:, 0]
        u = depth_pixel[:, 1]
        z = depth_pixel[:, 2]
        cx = self.cam_info.center_x()
        cy = self.cam_info.center_y()
        fx = self.cam_info.focal_x()
        fy = self.cam_info.focal_y()
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.c_[x, y, z]

    def project_pC_to_img(self, pC: np.ndarray) -> np.ndarray:
        """Project 3D points in camera frame back into pixel coordinates."""

        projected = self.cam_info.intrinsic_matrix() @ pC.T
        return np.round((projected[:2, :] / projected[2, :]).T).astype(np.int32)

    def update_camera_pos(self):
        self.X_WC = self.cam.body_pose_in_world_output_port().Eval(self.cam_context)
        return self.X_WC

    def update_camera_feed(self) -> None:
        depth_im_read = (
            self.station.GetOutputPort(f"camera{self.idx}.depth_image")
            .Eval(self.station_context)
            .data.squeeze()
        )
        self.depth_im = deepcopy(depth_im_read)
        self.depth_im[self.depth_im == np.inf] = 10.0
        self.rgb_im = (
            self.station.GetOutputPort(f"camera{self.idx}.rgb_image")
            .Eval(self.station_context)
            .data
        )
