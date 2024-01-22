from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn

from direct.preprocessor import Preprocessor, pose_inv
from rotation_predictor import RotationPredictorBase

ModelRotPrediction = torch.Tensor
PredictedRotation = float
PointCloud = np.ndarray


class SceneData(TypedDict):
    image_0: np.ndarray # (Batch, image_height, image_width)
    image_1: np.ndarray
    depth_0: np.ndarray
    depth_1: np.ndarray
    seg_0: np.ndarray
    seg_1: np.ndarray
    intrinsics_0: np.ndarray # (3,3)
    intrinsics_1: np.ndarray
    T_WC: np.ndarray # Extrinsics


class ProcessedData(TypedDict):
    pointcloud_0: PointCloud
    pointcloud_1: PointCloud


class PosePredictor(nn.Module):

    def __init__(self,
                 device: str = "cuda:0",
                 filter_pointcloud: bool = True,
                 filter_outliers_o3d: bool = False,
                 n_points: int = 2048) -> None:
        super().__init__()

        self.device = device
        self.preprocessor = Preprocessor(n_points, filter_pointcloud, filter_outliers_o3d)
        self.rot_model: RotationPredictorBase = None

    def _initialise_models(self):
        self.rot_model.load_weights()
        # self.rot_model.to(device=self.device)  # TODO figure this out
        self.rot_model.eval()

    def rotate_pointcloud(self, pcd: PointCloud, angle_z: float):
        # print(np.rad2deg(angle_z))
        R = np.eye(3)
        cosine = np.cos(angle_z)
        sine = np.sin(angle_z)
        R[0, 0] = cosine
        R[1, 1] = cosine
        R[0, 1] = -sine
        R[1, 0] = sine

        pcd[:3, :] = R @ pcd[:3, :]
        return R, pcd

    def find_translation(self, pcd0: PointCloud, pcd1: PointCloud) -> np.ndarray:
        pcd0_centre = np.mean(pcd0[:3, :], axis=1)
        pcd1_centre = np.mean(pcd1[:3, :], axis=1)
        return pcd1_centre - pcd0_centre

    def forward(self, data: SceneData):
        data.update(self.preprocessor(data))
        predicted_rot = self.rot_model(data)
        R_mtx, rotated_pcd0 = self.rotate_pointcloud(data["pointcloud_0"], predicted_rot)
        translation = self.find_translation(rotated_pcd0, data["pointcloud_1"])

        T_delta_base = np.eye(4)
        T_delta_base[:3, :3] = R_mtx
        T_delta_base[:3, 3] = translation

        T_delta_cam = pose_inv(data["T_WC"]) @ T_delta_base @ data["T_WC"]

        return T_delta_base, T_delta_cam #(4x4)
