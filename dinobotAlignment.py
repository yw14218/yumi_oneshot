
import numpy as np 
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings("ignore")
import math
import torch
from PIL import Image
#Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features
from DinoViT.correspondences import find_correspondences, draw_correspondences
from scipy.spatial.transform import Rotation
from trajectory_utils import create_homogeneous_matrix, pose_inv

class DINOBotAlignment:
    def __init__(self):
        # Hyperparameters for DINO correspondences extraction
        self.num_pairs = 8 #@param
        self.load_size = 224 #@param
        self.layer = 9 #@param
        self.facet = 'key' #@param
        self.bin = True #@param
        self.thresh = 0.05 #@param
        self.model_type = 'dino_vits8' #@param
        self.stride = 4 #@param
        self.camera_intrinsics = np.load("handeye/intrinsics_d405.npy")
        self.T_camera_eef = np.load("handeye/T_C_EEF_wrist_l.npy")
        self.show_plots = True

    @staticmethod
    def add_depth(points, depth_map, resized_shape):
        original_shape = depth_map.shape
        point_with_depth = []
        for (y, x) in points:
            x = int(x / resized_shape[0] * original_shape[0])
            y = int(y / resized_shape[1] * original_shape[1])
            point_with_depth.append([x, y, depth_map[y, x]])
        
        return point_with_depth
        
    def convert_pixels_to_meters(self, t):
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy =  self.camera_intrinsics[0, 2],  self.camera_intrinsics[1, 2]
        x = (t[0] - cx) * t[2] / fx
        y = (t[1] - cy) * t[2] / fy 
        
        return (x / 1000, y / 1000, t[2] / 1000)

    @staticmethod
    def find_transformation(X, Y):
        #Find transformation given two sets of correspondences between 3D points
        # Calculate centroids
        cX = np.mean(X, axis=0)
        cY = np.mean(Y, axis=0)
        # Subtract centroids to obtain centered sets of points
        Xc = X - cX
        Yc = Y - cY
        # Calculate covariance matrix
        C = np.dot(Xc.T, Yc)
        # Compute SVD
        U, S, Vt = np.linalg.svd(C)
        # Determine rotation matrix
        R = np.dot(Vt.T, U.T)
        # Determine translation vector
        t = cY - np.dot(R, cX)

        return R, t

    @staticmethod
    def compute_error(points1, points2):
        return np.linalg.norm(np.array(points1) - np.array(points2))

    def compute_new_eef_in_world(self, R, t, T_eef_world):
        delta_camera = create_homogeneous_matrix(t, R)
        return T_eef_world @ self.T_camera_eef @ delta_camera @ pose_inv(self.T_camera_eef)

    def run(self, rgb_live_path, depth_live_path, rgb_bn_path, depth_bn_path):
        depth_bn = np.array(Image.open(depth_bn_path))
        depth_live = np.array(Image.open(depth_live_path))
        with torch.no_grad():
            points1, points2, image1_pil, image2_pil = find_correspondences(rgb_bn_path, rgb_live_path, self.num_pairs, self.load_size, self.layer,
                                                                            self.facet, self.bin, self.thresh, self.model_type, self.stride)
        if self.show_plots:
            fig_1, ax1 = plt.subplots()
            ax1.axis('off')
            ax1.imshow(image1_pil)
            fig_2, ax2 = plt.subplots()
            ax2.axis('off')
            ax2.imshow(image2_pil)
            fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
            plt.show()

        # Given the pixel coordinates of the correspondences, add the depth channel
        resized_shape = np.array(image1_pil).shape
        points1 = self.add_depth(points1, depth_bn, resized_shape)
        points2 = self.add_depth(points2, depth_live, resized_shape)
        delta_R_camera, t = self.find_transformation(points1, points2)

        # A function to convert pixel distance into meters based on calibration of camera.
        delta_t_camera = self.convert_pixels_to_meters(t)
        error = self.compute_error(points1, points2)

        return delta_t_camera, delta_R_camera, error # delta_T in camera frame
