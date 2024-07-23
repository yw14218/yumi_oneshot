import numpy as np
import torch
import rospy
import ros_numpy
import time
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from DinoViT.hand_utils import extract_desc_maps, extract_descriptor_nn, draw_correspondences
from camera_utils import solve_transform_3d, d405_K as K, d405_T_C_EEF
from trajectory_utils import pose_inv, euler_from_quat, euler_from_matrix
from moveit_utils.cartesian_control import YuMiCartesianController
from lightglue import LightGlue, SuperPoint, match_pair, SIFT
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float, device="cuda")
    
class DINOBotVS:
    def __init__(self, DIR):
        # Hyperparameters for DINO correspondences extraction
        self.DIR = DIR

        self.num_pairs = 12 #@param
        self.load_size = 224 #@param
        self.rgb_ref = np.array(Image.open(f"{DIR}/demo_wrist_rgb.png"))
        self.depth_ref = np.array(Image.open(f"{DIR}/demo_wrist_depth.png"))
        self.seg_ref = np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool)

        self.width = self.depth_ref.shape[0]
        self.height = self.depth_ref.shape[1]

       # Load from file
        cache = np.load(f'{DIR}/dino.npz')

        self.desc1 = torch.tensor(cache['desc1'], device='cuda:0')
        self.descriptor_vectors = torch.tensor(cache['descriptor_vectors'], device='cuda:0')
        self.num_patches = tuple(cache['num_patches'])
        self.mkpts_0 = None
        self.index = 0
        self.cartesian_controller = YuMiCartesianController()
        self.dof = 3

        self.cap_t = 0.01
        self.cap_r = np.deg2rad(5)

        # self.extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
        # self.matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()

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

    def match_dino(self, live_image_dir, filter_seg=True):
        # Extract descriptors and original images
        descriptors_list, _ = extract_desc_maps([live_image_dir], load_size=self.load_size)
        
        if self.mkpts_0 is None:
            # Get keypoints for the reference image
            key_y, key_x = extract_descriptor_nn(self.descriptor_vectors, emb_im=self.desc1, 
                                                patched_shape=self.num_patches, return_heatmaps=False)
            mkpts_0 = np.array([(y * self.width / self.num_patches[0], x * self.height / self.num_patches[1]) 
                                for y, x in zip(key_y, key_x)])

            self.mkpts_0 = mkpts_0
        
        # Get keypoints for the live image
        key_y, key_x = extract_descriptor_nn(self.descriptor_vectors, emb_im=descriptors_list[0], 
                                            patched_shape=self.num_patches, return_heatmaps=False)
        mkpts_1 = np.array([(y * self.width / self.num_patches[0], x * self.height / self.num_patches[1]) 
                            for y, x in zip(key_y, key_x)])
    
        live_rgb = Image.open(live_image_dir)

        fig = draw_correspondences(self.mkpts_0, mkpts_1, Image.fromarray(self.rgb_ref), live_rgb, self.index)
        self.index += 1

        # Reverse x and y coordinates for both keypoints
        mkpts_0, mkpts_1 = self.mkpts_0[:, ::-1], mkpts_1[:, ::-1]

        if filter_seg:
            # Convert to integer coordinates for segmentation lookup
            coords = mkpts_0.astype(int)
            # Get segmentation values for corresponding coordinates
            seg_values = self.seg_ref[coords[:, 1], coords[:, 0]]
            # Filter points where segment values are True
            mask = seg_values

            mkpts_0 = mkpts_0[mask]
            mkpts_1 = mkpts_1[mask]
                    
        return mkpts_0, mkpts_1
    
    def match_siftlg(self, live_image_dir, filter_seg=True):
        rgb_live = np.array(Image.open(live_image_dir))
        feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, numpy_image_to_torch(self.rgb_ref), numpy_image_to_torch(rgb_live))
        matches = matches01['matches']  # indices with shape (K,2)
        mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
        mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)
        
        if filter_seg:
            # Convert to integer coordinates for segmentation lookup
            coords = mkpts_0.astype(int)
            # Get segmentation values for corresponding coordinates
            seg_values = self.seg_ref[coords[:, 1], coords[:, 0]]
            # Filter points where segment values are True
            mask = seg_values

            mkpts_0 = mkpts_0[mask]
            mkpts_1 = mkpts_1[mask]
                    
        return mkpts_0, mkpts_1
    
    def run(self):
        # Get the current pose and convert quaternion to Euler angles
        current_pose = yumi.get_current_pose(yumi.LEFT).pose
        current_rpy = np.array(euler_from_quat([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]))
        
        # Initialize error
        error = float('inf')
        start = time.time()

        while error > 0.005 and self.index < 150:
            
            # Save RGB and depth images
            live_rgb_dir, live_depth_dir = self.save_rgbd()
            
            # Match descriptors
            mkpts_0, mkpts_1 = self.match_dino(live_rgb_dir)
            # mkpts_0, mkpts_1 = self.match_siftlg(live_rgb_dir)

            if len(mkpts_0) <= 3:
                pass

            # Load current depth image
            depth_cur = np.array(Image.open(live_depth_dir))
            
            # Compute transformation
            T_delta_cam = solve_transform_3d(mkpts_0, mkpts_1, self.depth_ref, depth_cur, K)
            T_eef_world = yumi.get_curent_T_left()
            T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)
             # Update error
            error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            print(error)

            T_delta_eef = d405_T_C_EEF @ T_delta_cam @ pose_inv(d405_T_C_EEF)
            T_eef_world_new = T_eef_world @ T_delta_eef

            dx = T_eef_world_new[0, 3] - T_eef_world[0, 3]
            dy = T_eef_world_new[1, 3] - T_eef_world[1, 3]
            dz = T_eef_world_new[2, 3] - T_eef_world[2, 3]
            drz = euler_from_matrix(T_eef_world_new)[-1] - euler_from_matrix(T_eef_world)[-1]

            if abs(dx) > self.cap_t:
                dx = self.cap_t if dx > 0 else -self.cap_t
            if abs(dy) > self.cap_t:
                dy = self.cap_t if dy > 0 else -self.cap_t
            if abs(drz) > self.cap_r:
                drz = self.cap_r if drz > 0 else -self.cap_r

            if self.dof == 3:
                current_pose.position.x += dx
                current_pose.position.y += dy
                current_rpy[-1] += drz 

            eef_pose = yumi.create_pose_euler(current_pose.position.x, 
                                              current_pose.position.y, 
                                              current_pose.position.z, 
                                              current_rpy[0], 
                                              current_rpy[1], 
                                              current_rpy[2])
            
            self.cartesian_controller.move_eef(eef_pose)
        
        rospy.loginfo("DINO has aligned or max time allowed passed.")