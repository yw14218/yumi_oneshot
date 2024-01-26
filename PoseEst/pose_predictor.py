from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn

from direct.preprocessor import Preprocessor, pose_inv
from rotation_predictor import RotationPredictorBase
from direct.rotation_classificator import RotationClassificator

ModelRotPrediction = torch.Tensor
PredictedRotation = float
PointCloud = np.ndarray

def pose_inv(pose):
    R = pose[:3, :3]
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = - R.T @ np.ascontiguousarray(pose[:3, 3])
    return T

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
        self.rot_model: RotationPredictorBase = RotationClassificator()

    def _initialise_models(self):
        self.rot_model.load_weights()
        # self.rot_model.to(device=self.device)  # TODO figure this out
        print(self.rot_model.eval())

    def rotate_pointcloud(self, pcd: PointCloud, angle_z: float):
        print("predicted rotation", np.rad2deg(angle_z))
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
        print(f"After {np.random.randint(100)}")
        R_mtx, rotated_pcd0 = self.rotate_pointcloud(data["pointcloud_0"], predicted_rot)
        translation = self.find_translation(rotated_pcd0, data["pointcloud_1"])
        T_delta_base = np.eye(4)
        T_delta_base[:3, :3] = R_mtx
        T_delta_base[:3, 3] = translation

        T_delta_cam = pose_inv(data["T_WC"]) @ T_delta_base @ data["T_WC"]

        return T_delta_base, T_delta_cam #(4x4)

if __name__ == "__main__":

    from PIL import Image
    import open3d as o3d

    T_WC = np.load("../handeye/T_WC_head.npy")

    # demon_rgb = Image.open("../data/lego/demo_clipped_image.png")
    # demon_depth = Image.open("../data/lego/demo_clipped_depth.png")
    # demon_mask = Image.open("../data/lego/demo_clipped_mask.png")
    # live_rgb = Image.open("../data/lego/live_clipped_image.png")
    # live_depth = Image.open("../data/lego/live_clipped_depth.png")
    # live_mask = Image.open("../data/lego/live_clipped_mask.png")

    # demon_rgb = Image.open("../data/lego/original/demo_rgb.png")
    # demon_depth = Image.open("../data/lego/original/demo_depth.png")
    # demon_mask = Image.open("../data/lego/original/demo_mask.png")

    live_rgb = Image.open("../data/live_spray_rgb.png")
    live_depth = Image.open("../data/live_spray_depth.png")
    live_mask = Image.open("../data/live_spray_mask.png")

    intrinsics = np.load("../handeye/intrinsics.npy").reshape(3, 3)

    # demon_rgb_array = np.array(demon_rgb)
    # demon_depth_array = np.array(demon_depth)
    # demon_mask_array = np.array(demon_mask)

    # live_rgb_array = np.array(live_rgb)
    # live_depth_array = np.array(live_depth)
    # live_mask_array = np.array(live_mask)

    # np.save("live_rgb_array.npy", live_rgb_array)
    # np.save("live_depth_array.npy", live_depth_array)
    # np.save("live_mask_array.npy", live_mask_array)
    

    live_rgb_array = np.load("live_rgb_array.npy")
    live_depth_array = np.load("live_depth_array.npy")
    live_mask_array = np.load("live_mask_array.npy")

    # np.random.seed(10)
    # torch.random.manual_seed(10)


    # Create the SceneData instance with the loaded data
    data = SceneData(
        image_0=live_rgb_array,
        image_1=live_rgb_array,
        depth_0=live_depth_array,
        depth_1=live_depth_array,
        seg_0=live_mask_array,
        seg_1=live_mask_array,
        intrinsics_0=intrinsics,
        intrinsics_1=intrinsics,
        T_WC=T_WC
    )

    pose_predictor = PosePredictor()

    # # np.random.seed(10)
    # print(np.random.random_integers(100))
    T_delta_base, T_delta_cam = pose_predictor.forward(data)

    # print(T_delta_base)
    # print(T_delta_cam)

    # # new T_eef = T_delta_base @ T_eef

    import copy

    # def draw_registration_result(source, target, transformation):
    #     source.transform(transformation)
    #     source_temp = copy.deepcopy(source)
    #     target_temp = copy.deepcopy(target)
    #     source_temp.paint_uniform_color([1, 0.706, 0])
    #     target_temp.paint_uniform_color([0, 0.651, 0.929])
    #     o3d.visualization.draw_geometries([source_temp, target_temp])

    pcd0 = o3d.geometry.PointCloud()
    pcd1 = o3d.geometry.PointCloud()

    pcd0.points = o3d.utility.Vector3dVector(data["pc0"][:, :3])
    pcd1.points = o3d.utility.Vector3dVector(data["pc1"][:, :3])
    
    # draw_registration_result(source=pcd0, target=pcd1, transformation=T_delta_base)

    # # Extracting rotation matrix and translation vector
    # rotation_matrix = T_delta_base[:3, :3]
    # translation_vector = T_delta_base[:3, 3]

    import cv2
    # # Calculating rotation error
    # rvec, _ = cv2.Rodrigues(rotation_matrix)
    # rotation_error = np.linalg.norm(rvec) * 180 / np.pi

    # # Calculating translation error
    # translation_error = np.linalg.norm(translation_vector)

    # print(rotation_error, translation_error)

    # Function to draw registration results
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])


    # ICP registration
    threshold = 0.02  # Set a threshold for ICP, this depends on your data
    trans_init = np.identity(4)  # Initial transformation

    # Apply ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd0, pcd1, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Get the transformation matrix
    T_delta_icp = reg_p2p.transformation


    # Extracting rotation matrix and translation vector
    rotation_matrix = T_delta_icp[:3, :3]
    translation_vector = T_delta_icp[:3, 3]

    rvec, _ = cv2.Rodrigues(rotation_matrix)
    rotation_error = np.linalg.norm(rvec) * 180 / np.pi

    # Calculating translation error
    translation_error = np.linalg.norm(translation_vector)

    print(rotation_error, translation_error)

    # Draw the result
    draw_registration_result(pcd0, pcd1, T_delta_icp)
