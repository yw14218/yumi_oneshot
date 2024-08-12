import numpy as np
import rospy
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from camera_utils import solve_transform_3d, d405_K as K, d405_T_C_EEF as T_wristcam_eef
from trajectory_utils import pose_inv, euler_from_quat, state_to_transform, transform_to_state
from base_servoer import LightGlueVisualServoer
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter as UKF
from collections import deque

class GlobalMultiCamKFVisualServoing(LightGlueVisualServoer):
    def __init__(self, DIR, prior_state=None, prior_covariance=None):
        super().__init__(
            rgb_ref=np.array(Image.open(f"{DIR}/demo_wrist_rgb.png")),
            seg_ref=np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool),
            use_depth=True,
            features='sift'
        )
        self.DIR = DIR
        self.depth_ref = np.array(Image.open(f"{DIR}/demo_wrist_depth.png"))
        self.max_translation_step = 0.003
        self.max_rotation_step = np.deg2rad(3)
        self.moving_window = deque(maxlen=3) # sliding window of states to estimate variance
        # self.gains = [0.15, 0.15, 0.02, 0.1, 0.1, 0.2] # (proportional gain)
        self.gains = [0.15, 0.15, 0.15, 0.1, 0.1, 0.15] # (proportional gain)
        self.switch_threshold = (0.02, 10)  # (meters, degrees)

        self.prior_state = transform_to_state(prior_state) if prior_state is not None else None
        self.prior_covariance = prior_covariance
        self.ukf = self.initialize_ukf(prior_state=self.prior_state, prior_covariance=self.prior_covariance)

    def fx(self, x, dt):
        return x  # Identity function for state transition (no dynamics in this simple case)

    def hx(self, x):
        return x  # Identity function for measurement (we directly measure all states)
    
    def initialize_ukf(self, prior_state=None, prior_covariance=None):
        # Define the dimensionality of the state
        state_dim = 6  # [x, y, z, roll, pitch, yaw]
        sigma_points = MerweScaledSigmaPoints(n=state_dim, alpha=0.1, beta=2.0, kappa=3-state_dim)
        
        ukf = UKF(dim_x=state_dim, dim_z=state_dim, fx=self.fx, hx=self.hx, dt=1.0, points=sigma_points)

        if prior_state is None and prior_covariance is None:
            ukf.P = np.eye(state_dim) * 100000  # high initial uncertainty
        else:
            ukf.x = self.prior_state
            ukf.P = np.eye(state_dim) * 0.5

        ukf.R = np.eye(state_dim) * 10  # measurement noise
        ukf.Q = np.eye(state_dim) # zero process noise
        
        return ukf

    def add_measurements(self, state):
        self.moving_window.append(state)
        return np.cov(np.array(self.moving_window), rowvar=False) if len(self.moving_window) == 3 else None

    @staticmethod
    def is_measurement_valid(cov_matrix, trans_var_threshold=np.array([0.05, 0.05, 0.05])):   
        return np.all(cov_matrix[:3, :3].diagonal() < trans_var_threshold)

    @staticmethod
    def compute_goal_state(T_delta_cam):
        T_current_eef_world = yumi.get_curent_T_left()
        T_delta_eef = T_wristcam_eef @ T_delta_cam @ pose_inv(T_wristcam_eef)
        return transform_to_state(T_current_eef_world @ T_delta_eef), transform_to_state(T_current_eef_world)

    @staticmethod
    def state_to_T_delta_cam(state):
        T_current_eef_world = yumi.get_curent_T_left()
        filtered_T_delta_cam = pose_inv(T_wristcam_eef) @ pose_inv(T_current_eef_world) @ state_to_transform(state) @ T_wristcam_eef
        return filtered_T_delta_cam
    
    def compute_control_input(self, goal_state, current_state):
        """Compute control input based on the current state estimate."""
        translation = goal_state[:3] - current_state[:3]
        rotation = goal_state[3:] - current_state[3:]
        control_input = np.concatenate([
            np.array([np.clip(self.gains[0] * translation[0], -self.max_translation_step, self.max_translation_step)]),
            np.array([np.clip(self.gains[1] * translation[1], -self.max_translation_step, self.max_translation_step)]),
            np.array([np.clip(self.gains[2] * translation[2], -self.max_translation_step, self.max_translation_step)]),
            np.array([np.clip(self.gains[3] * rotation[0], -self.max_rotation_step, self.max_rotation_step)]),
            np.array([np.clip(self.gains[4] * rotation[1], -self.max_rotation_step, self.max_rotation_step)]),
            np.array([np.clip(self.gains[5] * rotation[2], -self.max_rotation_step, self.max_rotation_step)])
        ])
        return control_input
    
    def run_global(self):
        current_pose = yumi.get_current_pose(yumi.LEFT).pose
        current_rpy = np.array(euler_from_quat([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]))

        object_in_view_with_confidence = False

        while not object_in_view_with_confidence:
            mkpts_scores_0, mkpts_scores_1, depth_cur = self.match_lightglue(filter_seg=True)

            if mkpts_scores_0 is not None:
                print(len(mkpts_scores_0))

            if mkpts_scores_0 is not None and len(mkpts_scores_0) >= 10:
                T_delta_cam = solve_transform_3d(mkpts_scores_0[:, :2], mkpts_scores_1[:, :2], self.depth_ref, depth_cur, K, fall_back=False)
                if T_delta_cam is not None:
                    goal_state, current_state = self.compute_goal_state(T_delta_cam)
                    cov_matrix = self.add_measurements(goal_state)

                    print(cov_matrix)
                    if cov_matrix is not None:
                        object_in_view_with_confidence = self.is_measurement_valid(cov_matrix)
                        print(object_in_view_with_confidence)

            current_state = transform_to_state(yumi.get_curent_T_left())

            # Compute and apply control input
            control_input = self.compute_control_input(self.prior_state, current_state)

            current_pose.position.x += control_input[0]
            current_pose.position.y += control_input[1]
            current_pose.position.z += control_input[2]
            # current_rpy[0] += control_input[5]
            # current_rpy[1] += control_input[5]
            current_rpy[2] += control_input[5]

            # Create the next pose using updated position and orientation
            eef_pose_next = yumi.create_pose_euler(
                current_pose.position.x, 
                current_pose.position.y, 
                current_pose.position.z, 
                current_rpy[0], 
                current_rpy[1], 
                current_rpy[2]
            )
            
            # Move end effector to the next pose
            self.cartesian_controller.move_eef(eef_pose_next)
            

    def run(self):
        # self.run_global()

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
            self.ukf.predict()

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
            if translation_error < self.switch_threshold[0] and rotation_error < self.switch_threshold[1]:
                break

            # Compute the current state estimate
            goal_state, current_state = self.compute_goal_state(T_delta_cam)

            cov_matrix = self.add_measurements(goal_state)
            if cov_matrix is None:
                continue

            self.ukf.R = np.eye(6) * 10 + cov_matrix * 1000
            
            # UKF measurement update
            if self.is_measurement_valid(cov_matrix):
                self.ukf.update(goal_state)

            goal_state = self.ukf.x
            control_input = self.compute_control_input(goal_state, current_state)

            current_pose.position.x += control_input[0]
            current_pose.position.y += control_input[1]
            current_pose.position.z += control_input[2]
            # current_rpy[0] += control_input[3]
            # current_rpy[1] += control_input[4]
            current_rpy[2] += control_input[5]

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

        rospy.loginfo("Global alignment achieved, switching to refined local alignment.")

        return self.state_to_T_delta_cam(goal_state)