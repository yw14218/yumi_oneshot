import numpy as np
import rospy
import cv2
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from camera_utils import d405_K as K, d405_T_C_EEF as T_wristcam_eef, normalize_mkpts, model_selection_homography, weighted_solve_transform_3d
from trajectory_utils import pose_inv, euler_from_quat
from base_servoer import LightGlueVisualServoer
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter as UKF
from collections import deque
from scipy.spatial.transform import Rotation as R

class RefinedLocalVisualServoer(LightGlueVisualServoer):
    def __init__(self, DIR):
        super().__init__(
            rgb_ref=np.array(Image.open(f"{DIR}/demo_wrist_rgb.png")),
            seg_ref=np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool),
            use_depth=True,
            features='superpoint'
        )
        self.DIR = DIR
        self.depth_ref = np.array(Image.open(f"{DIR}/demo_wrist_depth.png"))
        self.gain = 0.05
        self.use_homography = None
        self.m_star = None
        self.Z_star = None
        self.previous_estimate_of_R = None

        self.pid_x = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
        self.pid_y = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
        self.pid_z = PIDController(Kp=0.025, Ki=0.0, Kd=0.005)
        self.pid_rx = PIDController(Kp=0.025, Ki=0.0, Kd=0.005)
        self.pid_ry = PIDController(Kp=0.025, Ki=0.0, Kd=0.005)
        self.pid_rz = PIDController(Kp=0.025, Ki=0.0, Kd=0.005)

    def decompose_homography(self, H_norm):
        # Decompose the homography matrix into possible rotation matrices, translation vectors, and normals
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H_norm, np.eye(3))

        best_solution_index = None
        best_solution_error = float('inf')
        
        # Iterate over the decompositions and select the best one based on specific criteria
        for i in range(num_solutions):
            R = rotations[i]
            t = translations[i]
            
            # Check if the Z-component of the translation vector is positive (positive depth)
            if t[2] <= 0:
                continue
            
            error = np.arccos((np.trace(np.dot(self.previous_estimate_of_R.T, R)) - 1) / 2)

            # Select the best solution based on score
            if error < best_solution_error:
                best_solution_error = error
                best_solution_index = i

        if best_solution_index is not None:
            selected_R = rotations[best_solution_index]
            selected_t = translations[best_solution_index]
            self.previous_estimate_of_R = selected_R
            return selected_R, selected_t.flatten()
        else:
            return None, None

    def update_reference_point(self, mkpts_0, mkpts_1, K, thresh=0.5):
        ransac_thr = thresh / np.mean([K[0, 0], K[1, 1], K[0, 0], K[1, 1]])
        normalized_mkpts_0 = normalize_mkpts(mkpts_0, K)
        normalized_mkpts_1 = normalize_mkpts(mkpts_1, K)
        
        try:
            # Step 1: Get homography
            H_norm, inlier_mask = cv2.findHomography(normalized_mkpts_0, normalized_mkpts_1, cv2.USAC_MAGSAC, ransac_thr, confidence=0.99999)
            # Step 2: Filter the Inlier Points
            inlier_mkpts_0 = normalized_mkpts_0[inlier_mask.ravel() == 1]
            inlier_mkpts_1 = normalized_mkpts_1[inlier_mask.ravel() == 1]

            # Convert to original coordinates to match with the depth map
            inlier_mkpts_0_orig = mkpts_0[inlier_mask.ravel() == 1]

            # Step 3: Filter inliers with non-zero depth and get their depth values
            inlier_coords = inlier_mkpts_0_orig[:, :2].astype(int)
            depth_values = self.depth_ref[inlier_coords[:, 1], inlier_coords[:, 0]]
            non_zero_mask = depth_values > 0

            if not np.any(non_zero_mask):
                raise ValueError("No inliers with non-zero depth found.")

            # Filter inliers and depth values based on non-zero depth
            inlier_mkpts_0_non_zero = inlier_mkpts_0[non_zero_mask]
            inlier_mkpts_1_non_zero = inlier_mkpts_1[non_zero_mask]

            if len(inlier_mkpts_0_non_zero) == 0:
                raise ValueError("No inliers with non-zero depth found.")
            
            depth_values_non_zero = depth_values[non_zero_mask]

            # Step 4: Calculate Reprojection Errors
            reprojected_points_homogeneous = (H_norm @ np.hstack([inlier_mkpts_0_non_zero]).T).T
            reprojected_points = reprojected_points_homogeneous[:, :2] / reprojected_points_homogeneous[:, 2].reshape(-1, 1)
            reprojection_errors = np.linalg.norm(reprojected_points - inlier_mkpts_1_non_zero[:, :2], axis=1)

            # Step 5: Identify the Most Confident Inlier and its Depth Value
            most_confident_inlier_idx = np.argmin(reprojection_errors)
            most_confident_inlier = inlier_mkpts_0_non_zero[most_confident_inlier_idx]
            most_confident_inlier_depth = depth_values_non_zero[most_confident_inlier_idx]

            self.m_star = most_confident_inlier
            self.Z_star = most_confident_inlier_depth/1000
            print("Most Confident Inlier Point with Non-Zero Depth:", self.m_star)
            print("Corresponding Depth Value:", self.Z_star)

        except Exception as e:
            print(f"An error occurred: {e}")


    def get_task_function_homography(self, mkpts_0, mkpts_1, depth_cur, K):
        # Normalize keypoints using camera intrinsics
        normalized_mkpts_0 = normalize_mkpts(mkpts_0, K)
        normalized_mkpts_1 = normalize_mkpts(mkpts_1, K)

        # Compute homography using normalized keypoints to improve robustness
        H_norm, inlier_mask = cv2.findHomography(normalized_mkpts_0, normalized_mkpts_1, cv2.USAC_MAGSAC, 0.001, confidence=0.99999)


        selected_R, selected_t = self.decompose_homography(H_norm)
        self.R_delta_cam_before_switch = selected_R
        
        r_theta = R.from_matrix(selected_R).as_euler('xyz')
        
        reprojected_points_homogeneous = H_norm @ self.m_star 
        m = reprojected_points_homogeneous / reprojected_points_homogeneous[2]

        pixel_coords = K @ m  # Matrix multiplication
        u, v = pixel_coords[:2]  # Extract u and v

        pixel_coords_ref = K @ self.m_star
        u_star, v_star = pixel_coords_ref[:2] 
        Z = depth_cur[int(v), int(u)] / 1000

        if Z == 0:
            # Define the 3x3 window around (u, v)
            u_min, u_max = max(0, int(u) - 1), min(depth_cur.shape[1] - 1, int(u) + 1)
            v_min, v_max = max(0, int(v) - 1), min(depth_cur.shape[0] - 1, int(v) + 1)
            
            # Extract the 3x3 window
            window = depth_cur[v_min:v_max + 1, u_min:u_max + 1]

            # Calculate the average depth of non-zero values in the window
            non_zero_values = window[window > 0]
            if non_zero_values.size > 0:
                Z = np.mean(non_zero_values) / 1000
            else:
                return None, None, None

        error = np.linalg.norm(np.array([u_star, v_star]) - np.array([u, v]), ord=1)
        # print(f"Pixel coordinates: u = {u}, v = {v}, Depth Z = {Z}, Error = {error}")

        pho = Z / self.Z_star
        delta_x = (self.m_star[0] - pho * m[0]) * self.Z_star
        delta_y = (self.m_star[1] - pho * m[1]) * self.Z_star
        
        delta_z = 1 - pho
        delta_t = [delta_x, delta_y, delta_z]
        delta_r = r_theta

        return delta_t, delta_r, error

    def get_task_function_3d(self, mkpts_scores_0, mkpts_scores_1, depth_cur, K):
        mkpts_0_3d_weighted, mkpts_1_3d_weighted, T_delta_cam = weighted_solve_transform_3d(mkpts_scores_0, mkpts_scores_1, self.depth_ref, depth_cur, K)
        R_delta_cam = T_delta_cam[:3, :3]

        # Select top 4 matches
        X = mkpts_0_3d_weighted[:4, :3]
        Y = mkpts_1_3d_weighted[:4, :3]
        weights = mkpts_0_3d_weighted[:4, 3] 

        # Calculate weighted centroids
        cX = np.average(X, axis=0, weights=weights)
        cY = np.average(Y, axis=0, weights=weights)

        delta_t = cX - cY
        delta_r = R.from_matrix(R_delta_cam).as_euler('xyz')
        error = np.linalg.norm(cX - cY, ord=1)

        return -delta_t, delta_r, error

    def run(self, T_delta_cam_before_switch, max_iterations=150):
        # Get the current pose and convert quaternion to Euler angles
        current_pose = yumi.get_current_pose(yumi.LEFT).pose
        current_rpy = np.array(euler_from_quat([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]))

        # Initialize error
        trajectories = []
        self.errors = []
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

            if self.use_homography is None:
                self.use_homography = model_selection_homography(mkpts_scores_0[:, :2] , mkpts_scores_1[:, :2], K)

            if not self.use_homography:
                delta_t, delta_r, error = self.get_task_function_3d(mkpts_scores_0, mkpts_scores_1, depth_cur, K)
                print(f"Step {iteration + 1}, Error is : {error:.4g}")
            else:
                if self.m_star is None:
                    self.update_reference_point(mkpts_scores_0[:, :2], mkpts_scores_1[:, :2], K)
                    self.previous_estimate_of_R = T_delta_cam_before_switch[:3, :3]
                delta_t, delta_r, error = self.get_task_function_homography(mkpts_scores_0[:, :2], mkpts_scores_1[:, :2], depth_cur, K)
                if delta_r is None:
                    continue
                print(f"Step {iteration + 1}, Pixel Error is : {error:.4g}")

            rotation_error = np.rad2deg(np.linalg.norm(delta_r, ord=1))
            print(f"Step {iteration + 1}, Depth Error is: {delta_t[2]:.4g}, Rotation Error is : {rotation_error:.4g}")

            if error < 0.3 and rotation_error < 0.5 and delta_t[2] < 0.002:
                break
            control_input_x = np.clip(self.pid_x.update(delta_t[0]), -0.002, 0.002)
            control_input_y = np.clip(self.pid_y.update(delta_t[1]), -0.002, 0.002)
            control_input_z = np.clip(self.pid_z.update(delta_t[2]), -0.002, 0.002)
            control_input_rx = np.clip(self.pid_rx.update(delta_r[0]), -0.05, 0.05)
            control_input_ry = np.clip(self.pid_ry.update(delta_r[1]), -0.05, 0.05)
            control_input_rz = np.clip(self.pid_rz.update(delta_r[2]), -0.05, 0.05)

            # Move robot by the updated control input
            current_pose.position.x += control_input_x
            current_pose.position.y -= control_input_y
            current_pose.position.z += control_input_z
            # current_rpy[0] -= control_input_rx
            # current_rpy[1] -= control_input_ry
            current_rpy[2] -= control_input_rz
            
            # trajectories.append([current_pose.position.x, current_pose.position.y, np.degrees(current_rpy[-1])])

            new_pose = yumi.create_pose_euler(current_pose.position.x, current_pose.position.y, current_pose.position.z, current_rpy[0], current_rpy[1], current_rpy[2])
            self.cartesian_controller.move_eef(new_pose)

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