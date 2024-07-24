import numpy as np
import rospy
import ros_numpy
import time
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from camera_utils import solve_transform_3d, d405_K as K, d405_T_C_EEF, normalize_mkpts
from trajectory_utils import pose_inv, euler_from_quat, euler_from_matrix
from moveit_utils.cartesian_control import YuMiCartesianController
from lightglue import SIFT, LightGlue
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from vis import visualize_convergence_on_sphere
from scipy.spatial.transform import rotation
import warnings
import matplotlib.pyplot as plt
import cv2
warnings.filterwarnings("ignore")


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    
class HomVS:
    def __init__(self, DIR):
        self.DIR = DIR

        self.depth_ref = np.array(Image.open(f"{DIR}/demo_wrist_depth.png"))
        self.seg_ref = np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool)

        self.width = self.depth_ref.shape[0]
        self.height = self.depth_ref.shape[1]

        self.index = 0
        self.cartesian_controller = YuMiCartesianController()
        self.dof = 3

        self.cap_t = 0.01
        self.cap_r = np.deg2rad(5)

        self.extractor = SIFT(backend='pycolmap', max_num_keypoints=1024).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher
        self.feats0 = self.extractor.extract(load_image(f"{DIR}/demo_wrist_rgb.png").cuda())

    def get_rgb(self):
        rgb_message_wrist = rospy.wait_for_message("d405/color/image_rect_raw", ImageMsg, timeout=5)
        return ros_numpy.numpify(rgb_message_wrist)
    
    def match_siftlg(self, live_image, filter_seg=True):
        live_image = numpy_image_to_torch(live_image)
        feats1 = self.extractor.extract(live_image)
        matches01 = self.matcher({'image0': self.feats0, 'image1': feats1})     
        feats0, feats1, matches01 = [rbd(x) for x in [self.feats0, feats1, matches01]]  # remove batch dimension
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
    
    @staticmethod
    def select_best_decomposition(decompositions):
        Rs, ts, normals = decompositions[1], decompositions[2], decompositions[3]
        best_index = -1
        best_score = float('inf')

        for i, (R, t, n) in enumerate(zip(Rs, ts, normals)):
            # # Heuristic 1: Check the magnitude of the translation vector (consider all components)
            # translation_magnitude = np.linalg.norm(t)

            # # Heuristic 2: Ensure the normal is close to [0, 0, 1]
            # normal_deviation = np.linalg.norm(n - np.array([0, 0, 1]))

            # Heuristic 3: Check the rotation matrix (for dominant yaw)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
            roll = np.arctan2(R[2, 1], R[2, 2])

            # Ideally, pitch and roll should be small in a top-down view
            rotation_score = np.abs(pitch) + np.abs(roll)

            # Combine heuristics into a single score
            score = rotation_score

            if score < best_score:
                best_score = score
                best_index = i

        if best_index != -1:
            best_R = Rs[best_index]
            best_t = ts[best_index]
            best_normal = normals[best_index]
        else:
            print("No valid decomposition found.")

        return best_R, best_t
    
    def run(self):
        # Get the current pose and convert quaternion to Euler angles
        current_pose = yumi.get_current_pose(yumi.LEFT).pose
        current_rpy = np.array(euler_from_quat([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]))
        
        d405_T_C_EEF[0, 3] = 0
        d405_T_C_EEF[1, 3] = 0
        d405_T_C_EEF[2 ,3] = 0

        pid_x = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
        pid_y = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
        pid_rz = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)

        # Initialize error
        error = float('inf')
        start = time.time()
        trajectory = []

        num_iteration = 0
        while error > 0.005 and num_iteration < 100:
            
            # Save RGB and depth images
            live_rgb = self.get_rgb()
            
            # Match descriptors
            mkpts_0, mkpts_1 = self.match_siftlg(live_rgb)

            if len(mkpts_0) <= 3:
                pass

            mkpts_0_norm = normalize_mkpts(mkpts_0, K)
            mkpts_1_norm = normalize_mkpts(mkpts_1, K)

            H_norm, _ = cv2.findHomography(mkpts_0_norm, mkpts_1_norm, cv2.USAC_MAGSAC, 0.001, confidence=0.99999)
            decompositions = cv2.decomposeHomographyMat(H_norm, np.eye(3))
            best_R, best_t = self.select_best_decomposition(decompositions)

            # Compute transformation
            T_delta_cam = np.eye(4)
            T_delta_cam[:3, :3] = best_R
            T_delta_cam[:3 ,3] = best_t[0]
            T_delta_cam[0, 3] = H_norm[0, 2]
            T_delta_cam[1, 3] = H_norm[1, 2]

            T_eef_world = yumi.get_curent_T_left()
            T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)

             # Update error
            error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            print(error)
            T_delta_eef = d405_T_C_EEF @ T_delta_cam @ pose_inv(d405_T_C_EEF)
            T_eef_world_new = T_eef_world @ T_delta_eef

            dx = T_eef_world_new[0, 3] - T_eef_world[0, 3]
            dy = T_eef_world_new[1, 3] - T_eef_world[1, 3]
            drz = euler_from_matrix(T_eef_world_new)[-1] - euler_from_matrix(T_eef_world)[-1]

            dx = pid_x.update(dx)
            dy = pid_y.update(dy)
            drz = pid_rz.update(drz)

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
            trajectory.append([current_pose.position.x, current_pose.position.y, np.degrees(current_rpy[-1])])
            num_iteration += 1
        
        rospy.loginfo("DINO has aligned or max time allowed passed.")
        visualize_convergence_on_sphere(np.array(trajectory))