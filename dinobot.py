import numpy as np
import torch
import rospy
import ros_numpy
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from DinoViT.hand_utils import extract_descriptors, extract_desc_maps, extract_descriptor_nn
from camera_utils import solve_transform_3d, d405_K as K
from trajectory_utils import pose_inv, euler_from_quat, euler_from_matrix
from moveit_utils.cartesian_control import YuMiCartesianController
from experiments import load_experiment
import warnings
warnings.filterwarnings("ignore")

class DINOBotVS:
    def __init__(self, DIR):
        # Hyperparameters for DINO correspondences extraction
        self.DIR = DIR

        self.num_pairs = 12 #@param
        self.load_size = 224 #@param

        self.depth_ref = np.array(Image.open(f"{DIR}/demo_wrist_depth.png"))
        self.seg_ref = np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool)

        self.width = self.depth_ref.shape[0]
        self.height = self.depth_ref.shape[1]

       # Load from file
        cache = np.load(f'{DIR}/dino.npz')

        self.desc1 = torch.tensor(cache['desc1'], device='cuda:0')
        self.descriptor_vectors = torch.tensor(cache['descriptor_vectors'], device='cuda:0')
        self.num_patches = tuple(cache['num_patches'])

        self.cartesian_controller = YuMiCartesianController()

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
        
        # Get keypoints for the reference image
        key_y, key_x = extract_descriptor_nn(self.descriptor_vectors, emb_im=self.desc1, 
                                            patched_shape=self.num_patches, return_heatmaps=False)
        mkpts_0 = np.array([(y * self.width / self.num_patches[0], x * self.height / self.num_patches[1]) 
                            for y, x in zip(key_y, key_x)])
        
        # Get keypoints for the live image
        key_y, key_x = extract_descriptor_nn(self.descriptor_vectors, emb_im=descriptors_list[0], 
                                            patched_shape=self.num_patches, return_heatmaps=False)
        mkpts_1 = np.array([(y * self.width / self.num_patches[0], x * self.height / self.num_patches[1]) 
                            for y, x in zip(key_y, key_x)])
        
        # Reverse x and y coordinates for both keypoints
        mkpts_0, mkpts_1 = mkpts_0[:, ::-1], mkpts_1[:, ::-1]
        

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
        
        while error > 0.005:
            # Save RGB and depth images
            live_rgb_dir, live_depth_dir = self.save_rgbd()
            
            # Match descriptors
            mkpts_0, mkpts_1 = self.match_dino(live_rgb_dir)
            
            if len(mkpts_0) <= 3:
                pass

            # Load current depth image
            depth_cur = np.array(Image.open(live_depth_dir))
            
            # Compute transformation
            T_est = solve_transform_3d(mkpts_0, mkpts_1, self.depth_ref, depth_cur, K)
            delta_T = np.eye(4) @ pose_inv(T_est)
            
            # Update error
            error = np.linalg.norm(delta_T[:3, 3])
            print(error)

            # Update pose
            current_pose.position.x += delta_T[0, 3]
            current_pose.position.y -= delta_T[1, 3]
            
            # Extract and update roll, pitch, yaw (Euler angles)
            _, _, rz = euler_from_matrix(T_est)
            # current_rpy[-1] -= rz

            eef_pose = yumi.create_pose_euler(current_pose.position.x, 
                                              current_pose.position.y, 
                                              current_pose.position.z, 
                                              current_rpy[0], 
                                              current_rpy[1], 
                                              current_rpy[2])
            self.cartesian_controller.move_eef(eef_pose)
        
        rospy.loginfo("DINO has aligned.")
        
