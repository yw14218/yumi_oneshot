import numpy as np
import rospy
import ros_numpy
import time
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from camera_utils import solve_transform_3d, normalize_mkpts, d405_K as K, d405_T_C_EEF as T_wristcam_eef, d415_T_WC as T_WC, compute_homography
from trajectory_utils import pose_inv, euler_from_quat, state_to_transform, transform_to_state
from vis import visualize_convergence_on_sphere
import warnings
import matplotlib.pyplot as plt
from base_servoer import LightGlueVisualServoer, Dust3RisualServoer
from experiments import load_experiment
warnings.filterwarnings("ignore")
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from collections import deque

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
        self.dof = 3
        self.max_translation_step = 0.005
        self.max_rotation_step = np.deg2rad(5)
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
        return np.cov(np.array(self.moving_window), rowvar=False) if len(self.moving_window) == 3 else None

    @staticmethod
    def is_measurement_valid(cov_matrix, trans_var_threshold=np.array([0.005, 0.005, 0.005])):   
        return np.all(cov_matrix[:3, :3].diagonal() < trans_var_threshold)

    @staticmethod
    def compute_goal_state(T_delta_cam):
        T_current_eef_world = yumi.get_curent_T_left()
        T_delta_eef = T_wristcam_eef @ pose_inv(T_delta_cam) @ pose_inv(T_wristcam_eef)
        return transform_to_state(T_current_eef_world @ T_delta_eef), transform_to_state(T_current_eef_world)

    def state_to_T_delta_cam(self, state):
        T_current_eef_world = yumi.get_curent_T_left()
        T_delta_cam = pose_inv(T_wristcam_eef) @ pose_inv(state_to_transform(state)) @ T_current_eef_world @ T_wristcam_eef
        return T_delta_cam
    
    def compute_control_input(self, goal_state, current_state, gain=0.05):
        """Compute control input based on the current state estimate."""
        translation = goal_state[:3] - current_state[:3]
        rotation = goal_state[3:] - current_state[3:]
        control_input = np.concatenate([
            np.clip(gain * translation, -self.max_translation_step, self.max_translation_step),
            np.clip(gain * rotation, -self.max_rotation_step, self.max_rotation_step)
        ])
        return control_input
    
    def run(self, switch_threshold = (0.01, 5)):
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
            T_delta_cam = solve_transform_3d(mkpts_scores_0[:, :2] , mkpts_scores_1[:, :2] , self.depth_ref, depth_cur, K)

            # Update error
            T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)
            translation_error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            rotation_error = np.rad2deg(np.arccos((np.trace(T_delta_cam_inv[:3, :3]) - 1) / 2))
            print(f"Translation Error: {translation_error:.6f}, Rotation Error: {rotation_error:.2f} degrees")
            if translation_error < switch_threshold[0] and rotation_error < switch_threshold[1]:
                break

            # Compute the current state estimate
            goal_state, current_state = self.compute_goal_state(T_delta_cam)
            
            cov_matrix = self.add_measurements(goal_state)
            if cov_matrix is None:
                continue

            print(self.is_measurement_valid(cov_matrix))
            
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
            
            num_iteration += 1

        rospy.loginfo("Global alignment achieved, switching to local alignment.")

        return self.state_to_T_delta_cam(goal_state)