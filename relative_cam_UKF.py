import numpy as np
import rospy
import ros_numpy
import time
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from camera_utils import solve_transform_3d, normalize_mkpts, d405_K as K, d405_T_C_EEF as T_wristcam_eef, d415_T_WC as T_WC, compute_homography
from trajectory_utils import pose_inv, euler_from_quat, state_to_transform, transform_to_state
from moveit_utils.cartesian_control import YuMiLeftArmCartesianController
from lightglue import SIFT, LightGlue
from lightglue.utils import load_image, rbd
from vis import visualize_convergence_on_sphere
import warnings
import matplotlib.pyplot as plt
from base_servoer import LightGlueVisualServoer, Dust3RisualServoer
from experiments import load_experiment
warnings.filterwarnings("ignore")
# from poseEstimation import PoseEstimation
# import argparse
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from collections import deque

class _HierachicalVS():
    def __init__(self, dir):
        self.dir = dir

    def run(self):
        # pbvskf = _PVBSKF(self.dir)
        # pbvskf.run()

        # del pbvskf
        _25dvs = _25DVS(self.dir)
        _25dvs.run()

    
class _PVBSKF(LightGlueVisualServoer):
    def __init__(self, DIR):
        super().__init__(
            rgb_ref=np.array(Image.open(f"{DIR}/demo_wrist_rgb.png")),
            seg_ref=np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool),
            use_depth=True,
            features='sift'
        )
        self.DIR = DIR
        self.depth_ref = np.array(Image.open(f"{DIR}/demo_wrist_depth.png"))

        self.cartesian_controller = YuMiLeftArmCartesianController()
        self.dof = 3

        self.max_translation_step = 0.005
        self.max_rotation_step = np.deg2rad(5)
        self.gain = 0.05

        self.ukf = self.initialize_ukf()
        self.moving_window = deque(maxlen=3)

    def fx(state, dt, control_input):
        """Predicts the next state given the current state and control input."""
        return state

    def hx(state):
        """Measurement model directly returns the state as we assume full observation."""
        return state
    
    def initialize_ukf(self):
        # Define the dimensionality of the state
        state_dim = 6  # [x, y, z, roll, pitch, yaw]
        sigma_points = MerweScaledSigmaPoints(n=state_dim, alpha=0.1, beta=2.0, kappa=3-state_dim)
        
        ukf = UKF(dim_x=state_dim, dim_z=state_dim, fx=self.fx, hx=self.hx, dt=1, points=sigma_points)
        
        # Initial state covariance matrix
        ukf.P = np.eye(state_dim) * 1  # initial uncertainty
        ukf.R = np.eye(state_dim) * 0.05  # measurement noise
        ukf.Q = np.eye(6) # zero process noise
        
        return ukf

    def add_measurements(self, state):
        self.moving_window.append(state)
        if len(self.moving_window) == 3:
            covariance_matrix = np.cov(np.array(self.moving_window), rowvar=False)
            return covariance_matrix
        else:
            return None

    def is_measurement_valid(self, covariance_matrix, translation_variance_threshold = np.array([0.005, 0.005, 0.005])):
        for i in range(3):  # Only check the translations
            if covariance_matrix[i, i] > translation_variance_threshold[i]:
                return False
        
        return True

    def compute_goal_state(self, T_delta_cam):
        T_delta_eef = T_wristcam_eef @ pose_inv(T_delta_cam) @ pose_inv(T_wristcam_eef)
        T_current_eef_world = yumi.get_curent_T_left()
        return transform_to_state(T_current_eef_world @ T_delta_eef), transform_to_state(T_current_eef_world)

    def compute_control_input(self, goal_state, current_state):
        """Compute control input based on the current state estimate."""
        translation = goal_state[:3] - current_state[:3]
        rotation = goal_state[3:] - current_state[3:]
        control_input = np.concatenate([
            np.clip(self.gain * translation, -self.max_translation_step, self.max_translation_step),
            np.clip(self.gain * rotation, -self.max_rotation_step, self.max_rotation_step)
        ])
        return control_input
    
    def run(self, switch_threshold = [0.01, 5]):
        # Get the current pose and convert quaternion to Euler angles
        current_pose = yumi.get_current_pose(yumi.LEFT).pose
        current_rpy = np.array(euler_from_quat([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]))

        trajectories = []
        states = []

        translation_error = 1e6
        rotation_error = 1e6
        num_iteration = 0

        trajectories.append([current_pose.position.x, current_pose.position.y, current_pose.position.z, np.degrees(current_rpy[-1])])
        rospy.sleep(0.1)

        while True:
            # self.ukf.predict()

            # 1. Get new measurement
            mkpts_scores_0, mkpts_scores_1, depth_cur = self.match_lightglue(filter_seg=True)
            if mkpts_scores_0 is None or len(mkpts_scores_0) <= 3:
                continue

            # Compute transformation
            T_delta_cam = solve_transform_3d(mkpts_scores_0[:, :2] , mkpts_scores_1[:, :2] , self.depth_ref, depth_cur, K, compute_homography=False)

            # Update error
            T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)
            translation_error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            rotation_error = np.rad2deg(np.arccos((np.trace(T_delta_cam_inv[:3, :3]) - 1) / 2))
            print(f"Translation Error: {translation_error:.6f}, Rotation Error: {rotation_error:.2f} degrees")
            if translation_error < switch_threshold[0] and rotation_error < switch_threshold[1]:
                break

            # Compute the current state estimate
            goal_state, current_state = self.compute_goal_state(T_delta_cam)
            
            covariance_matrix = self.add_measurements(goal_state)
            if covariance_matrix is None:
                continue

            print(self.is_measurement_valid(covariance_matrix))
            
            # UKF measurement update
            # self.ukf.update(current_state)
            # current_state = self.ukf.x
            control_input = self.compute_control_input(goal_state, current_state)

            current_pose.position.x -= control_input[0]
            current_pose.position.y -= control_input[1]
            current_pose.position.z -= control_input[2]
            # current_rpy[0] -= control_input[3]
            # current_rpy[1] -= control_input[4]
            current_rpy[2] -= control_input[5]

            eef_pose_next = yumi.create_pose_euler(current_pose.position.x, 
                                                   current_pose.position.y, 
                                                   current_pose.position.z, 
                                                   current_rpy[0], 
                                                   current_rpy[1], 
                                                   current_rpy[2])
            
            self.cartesian_controller.move_eef(eef_pose_next)
            
            trajectories.append([current_pose.position.x, current_pose.position.y, current_pose.position.z, np.degrees(current_rpy[-1])])
            states.append(current_state)
            
            # Update error
            T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)
            translation_error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            rotation_error = np.rad2deg(np.arccos((np.trace(T_delta_cam_inv[:3, :3]) - 1) / 2))
            print(f"Translation Error: {translation_error:.6f}, Rotation Error: {rotation_error:.2f} degrees")
            
            num_iteration += 1

        rospy.loginfo("Coarse alignment achieved, switching to fine alignment.")

class _25DVS(LightGlueVisualServoer):
    def __init__(self, DIR):
        super().__init__(
            rgb_ref=np.array(Image.open(f"{DIR}/demo_wrist_rgb.png")),
            seg_ref=np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool),
            use_depth=True,
            features='superpoint'
        )
        self.DIR = DIR
        self.depth_ref = np.array(Image.open(f"{DIR}/demo_wrist_depth.png"))

        self.cartesian_controller = YuMiLeftArmCartesianController()
        self.dof = 3

        self.gain = 0.05

    def decompose_homography(self, H_norm):
        # Decompose the homography matrix into possible rotation matrices, translation vectors, and normals
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H_norm, np.eye(3))

        best_solution_index = None
        best_solution_score = float('inf')

        # Iterate over the decompositions and select the best one based on specific criteria
        for i in range(num_solutions):
            R = rotations[i]
            t = translations[i]
            
            # Check if the Z-component of the translation vector is positive (positive depth)
            if t[2] <= 0:
                continue

            # Calculate yaw (rotation around the Z-axis)
            yaw = np.arctan2(R[1, 0], R[0, 0])

            # Calculate the score (penalize less yaw or negative depth)
            score = abs(yaw)

            # Select the best solution based on score
            if score < best_solution_score:
                best_solution_score = score
                best_solution_index = i

        if best_solution_index is not None:
            selected_R = rotations[best_solution_index]
            selected_t = translations[best_solution_index]

            return selected_R, selected_t.flatten()
        else:
            print("No valid solution found after homography decomposition.")

    def get_homography_point(self, mkpts_0, mkpts_1, depth_ref, K):
        normalized_mkpts_0 = normalize_mkpts(mkpts_0, K)
        normalized_mkpts_1 = normalize_mkpts(mkpts_1, K)

        # Compute homography using normalized keypoints to improve robustness
        H_norm, inlier_mask = cv2.findHomography(normalized_mkpts_0, normalized_mkpts_1, cv2.USAC_MAGSAC, 0.001, confidence=0.99999)
        inliers_0 = normalized_mkpts_0[inlier_mask.ravel() == 1]

        

    def get_homography_task_function(self, mkpts_0, mkpts_1, depth_ref, K):
        # Normalize keypoints using camera intrinsics
        normalized_mkpts_0 = normalize_mkpts(mkpts_0, K)
        normalized_mkpts_1 = normalize_mkpts(mkpts_1, K)

        # Compute homography using normalized keypoints to improve robustness
        H_norm, inlier_mask = cv2.findHomography(normalized_mkpts_0, normalized_mkpts_1, cv2.USAC_MAGSAC, 0.001, confidence=0.99999)
        
        selected_R, selected_t = self.decompose_homography(H_norm)
        delta_R = R.from_matrix(selected_R).as_euler('xyz')

        # Extract inlier points
        inliers_0 = normalized_mkpts_0[inlier_mask.ravel() == 1]
        
        if len(inliers_0) == 0:
            return None, None, None, None

        # Select the inlier point with the highest score (since sorted in descending order already)
        m_homogeneous = inliers_0[0]
        u_v = K @ m_homogeneous
        u_v /= u_v[2]  # Normalize by the last coordinate to get (u, v)
        u, v = int(u_v[0]), int(u_v[1])
        
        H = K @ H_norm @ np.linalg.inv(K)

        cur_pixel = H @ u_v

        # Get depth at the corresponding pixel location
        depth = depth_ref[v, u] / 1000.0 

        # Compute the transformed point in normalized coordinates
        transformed_m =  m_homogeneous - H_norm @ m_homogeneous
        transformed_m /= transformed_m[2]
        transformed_m *= depth

        # Compute the difference in 3D coordinates
        delta_t = m_homogeneous[:3] - transformed_m[:3]

        return delta_t, delta_R, u_v, cur_pixel

    def run(self, max_iterations=150, threshold=0.1):
        # Get the current pose and convert quaternion to Euler angles
        current_pose = yumi.get_current_pose(yumi.LEFT).pose
        current_rpy = np.array(euler_from_quat([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]))

        pid_x = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
        pid_y = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
        pid_z = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
        pid_rz = PIDController(Kp=0.1, Ki=0.0, Kd=0.02)

        # Initialize error
        trajectories = []
        states = []
        errors = []
        translation_error = 1e6
        rotation_error = 1e6
        num_iteration = 0

        trajectories.append([current_pose.position.x, current_pose.position.y, current_pose.position.z, np.degrees(current_rpy[-1])])
        rospy.sleep(0.1)

        for iteration in range(max_iterations):

            # 1. Get new measurement
            mkpts_scores_0, mkpts_scores_1, depth_cur = self.match_lightglue(filter_seg=True)
            if mkpts_scores_0 is None or len(mkpts_scores_0) <= 3:
                continue

            # Sort the keypoints by their scores (in descending order)
            sorted_indices = np.argsort(-mkpts_scores_0[:, -1])
            mkpts_scores_0 = mkpts_scores_0[sorted_indices]
            mkpts_scores_1 = mkpts_scores_1[sorted_indices]

            # Compute transformation
            # T_delta_cam, _ = solve_transform_3d(mkpts_scores_0[:, :2] , mkpts_scores_1[:, :2] , self.depth_ref, depth_cur, K, compute_homography=False)
            
            # Capture current image and detect projection pixel
            delta_t, delta_R, ref_pixel, cur_pixel  = self.get_homography_task_function(mkpts_scores_0[:, :2], mkpts_scores_1[:, :2], depth_cur, K)
            if delta_t is None:
                continue

            delta_x = ref_pixel[0] - cur_pixel[0]
            delta_y = ref_pixel[1] - cur_pixel[1]
            error = np.linalg.norm(np.array([ref_pixel]) - np.array(cur_pixel), ord=1)

            # Check if error is within threshold
            if (abs(delta_x) < threshold and abs(delta_y) < threshold and error < 1) or iteration == max_iterations - 1:
                print(abs(delta_x), abs(delta_y))
                break

            errors.append(error)

            control_input_x = pid_x.update(delta_t[0])
            control_input_y = pid_y.update(delta_t[1])
            # control_input_z = pid_z.update(delta_Z)
            control_input_rz = pid_rz.update(delta_R[-1])

            control_input_x = np.clip(control_input_x, -0.002, 0.002)
            control_input_y = np.clip(control_input_y, -0.002, 0.002)
            control_input_rz = np.clip(control_input_rz, -0.02, 0.02)

            rospy.loginfo(f"Step {iteration + 1}, Error is : {error:.4g}, delta_x: {control_input_x:.4g}, delta_y: {control_input_y:.4g}")

            # Move robot by the updated control input
            current_pose.position.x += control_input_x
            current_pose.position.y -= control_input_y
            # current_pose.position.z += control_input_z
            current_rpy[-1] -= control_input_rz
            
            trajectories.append([current_pose.position.x, current_pose.position.y, np.degrees(current_rpy[-1])])

            new_pose = yumi.create_pose_euler(current_pose.position.x, current_pose.position.y, current_pose.position.z, current_rpy[0], current_rpy[1], current_rpy[2])

            self.cartesian_controller.move_eef(new_pose)

        rospy.loginfo("Coarse alignment achieved, switching to fine alignment.")

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