import numpy as np
import rospy
import ros_numpy
import time
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from camera_utils import solve_transform_3d, normalize_mkpts, d405_K as K, d405_T_C_EEF as T_wristcam_eef, d415_T_WC as T_WC, homography_test
from trajectory_utils import pose_inv, euler_from_quat, state_to_transform, transform_to_state
from moveit_utils.cartesian_control import YuMiLeftArmCartesianController
from lightglue import SIFT, LightGlue
from lightglue.utils import load_image, rbd
from vis import visualize_convergence_on_sphere
import warnings
import matplotlib.pyplot as plt
from base_servoer import PIDController, LightGlueVisualServoer, Dust3RisualServoer
from experiments import load_experiment
warnings.filterwarnings("ignore")
# from poseEstimation import PoseEstimation
# import argparse
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
from scipy.spatial.transform import Rotation as R

class RelCamVSKF(LightGlueVisualServoer):
    def __init__(self, DIR):
        super().__init__(
            rgb_ref=np.array(Image.open(f"{DIR}/demo_wrist_rgb.png")),
            seg_ref=np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool),
            use_depth=True
        )
        self.DIR = DIR
        self.depth_ref = np.array(Image.open(f"{DIR}/demo_wrist_depth.png"))
        self.index = 0
        self.cartesian_controller = YuMiLeftArmCartesianController()
        self.dof = 3

        self.max_translation_step = 0.005
        self.max_rotation_step = np.deg2rad(5)

        self.gain = 0.05

        self.ukf = self.initialize_ukf()

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

    def compute_state(self, T_delta_cam):
        T_delta_eef = T_wristcam_eef @ T_delta_cam @ pose_inv(T_wristcam_eef)
        T_current_eef_world = yumi.get_curent_T_left()
        return transform_to_state(T_current_eef_world @ T_delta_eef), transform_to_state(T_current_eef_world)

    def compute_control_input(self, state, current_eef_world):
        """Compute control input based on the current state estimate."""
        translation = state[:3] - current_eef_world[:3]
        rotation = state[3:] - current_eef_world[3:]
        control_input = np.concatenate([
            np.clip(self.gain * translation, -self.max_translation_step, self.max_translation_step),
            np.clip(self.gain * rotation, -self.max_rotation_step, self.max_rotation_step)
        ])
        return control_input
    
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
        trajectories = []
        states = []
        translation_error = 1e6
        rotation_error = 1e6
        num_iteration = 0

        trajectories.append([current_pose.position.x, current_pose.position.y, current_pose.position.z, np.degrees(current_rpy[-1])])
        rospy.sleep(0.2)

        control_input = np.zeros(6)
        while num_iteration < 120:
            # self.ukf.predict()

            # 1. Get new measurement
            mkpts_0, mkpts_1, depth_cur, highest_confidence_index = self.match_lightglue(filter_seg=True, feature='sift')
            if mkpts_0 is None or len(mkpts_0) <= 3:
                continue
            # Compute transformation
            T_delta_cam, _ = solve_transform_3d(mkpts_0, mkpts_1, self.depth_ref, depth_cur, K, compute_homography=False)

            # Compute the current state estimate
            current_state, current_eef_world = self.compute_state(T_delta_cam)
            
            # UKF measurement update
            # self.ukf.update(current_state)
            # current_state = self.ukf.x
            control_input = self.compute_control_input(current_state, current_eef_world)

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
            
            # Update error
            T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)
            translation_error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            rotation_error = np.rad2deg(np.arccos((np.trace(T_delta_cam_inv[:3, :3]) - 1) / 2))
            num_iteration += 1
            print(f"Translation Error: {translation_error:.6f}, Rotation Error: {rotation_error:.2f} degrees")
        np.save("states_no_filter1.npy", np.array(states))
        rospy.loginfo("Coarse alignment achieved, switching to fine alignment.")

        # while True:
        #     mkpts_0, mkpts_1, depth_cur, highest_confidence_index = self.match_lightglue(filter_seg=True, feature='superpoint')
        #     # Compute transformation
        #     T_delta_cam, H_norm = solve_transform_3d(mkpts_0, mkpts_1, self.depth_ref, depth_cur, K, compute_homography=True)
            

        # visualize_convergence_on_sphere(np.array(trajectory))
