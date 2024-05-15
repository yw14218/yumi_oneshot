
import numpy as np 
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings("ignore")
import torch
import rospy
import ros_numpy
from PIL import Image
#Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features
from DinoViT.correspondences import find_correspondences, draw_correspondences
from trajectory_utils import pose_inv, translation_from_matrix, quaternion_from_matrix
from sensor_msgs.msg import Image as ImageMsg

class DINOBotAlignment:
    def __init__(self, DIR):
        # Hyperparameters for DINO correspondences extraction
        self.DIR = DIR
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
        self.error_threshold = 0.02

    def save_rgbd(self):
        rgb_message_wrist = rospy.wait_for_message("d405/color/image_rect_raw", ImageMsg, timeout=5)
        depth_message_wrist = rospy.wait_for_message("d405/aligned_depth_to_color/image_raw", ImageMsg, timeout=5)
        rgb_image_wrist = Image.fromarray(ros_numpy.numpify(rgb_message_wrist))
        depth_image_wrist = Image.fromarray(ros_numpy.numpify(depth_message_wrist))
        rgb_dir = f"{self.DIR}/live_wrist_rgb.png"
        depth_dir = f"{self.DIR}/live_wrist_depth.png"
        rgb_image_wrist.save(rgb_dir)
        depth_image_wrist.save(depth_dir)

        return rgb_dir, depth_dir
    
    def add_depth(self, points, depth_map, resized_shape):
        original_shape = depth_map.shape
        points_3d = []
        for (y, x) in points:
            x = int(x / resized_shape[0] * original_shape[0])
            y = int(y / resized_shape[1] * original_shape[1])
            points_3d.append(self.convert_pixels_to_meters([x, y, depth_map[y, x]]))
        
        return points_3d
        
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
        delta_camera = np.eye(4) 
        delta_camera[:3, :3] = R
        delta_camera[:3, 3] = t 
        T_new_eef_world = T_eef_world @ self.T_camera_eef @ delta_camera @ pose_inv(self.T_camera_eef)
        xyz = translation_from_matrix(T_new_eef_world).tolist()
        quaternion = quaternion_from_matrix(T_new_eef_world).tolist()

        return xyz + quaternion

    def run(self, rgb_live_path, depth_live_path):
        rgb_bn_path = "{0}/demo_wrist_rgb.png".format(self.DIR) 
        depth_bn_path = "{0}/demo_wrist_depth.png".format(self.DIR)
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
        points1_3d = self.add_depth(points1, depth_bn, resized_shape)
        points2_3d = self.add_depth(points2, depth_live, resized_shape)
        delta_R_camera, delta_t_camera = self.find_transformation(points1_3d, points2_3d)
        error = self.compute_error(points1_3d, points2_3d)

        rospy.loginfo(f'Error is {error}, while the stopping threshold is {self.thresh}.')
        return delta_t_camera, delta_R_camera, error # delta_T in camera frame
