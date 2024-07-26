import numpy as np
import rospy
import time
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from camera_utils import d405_K as K, d405_T_C_EEF, normalize_mkpts
from trajectory_utils import pose_inv, euler_from_quat, euler_from_matrix
from moveit_utils.cartesian_control import YuMiLeftArmCartesianController
from vis import visualize_convergence_on_sphere
from base_servoer import SiftLightGlueVisualServoer, PIDController
import warnings
import cv2
warnings.filterwarnings("ignore")
    
class HomVS(SiftLightGlueVisualServoer):
    def __init__(self, DIR):
        super().__init__(
            rgb_ref=np.array(Image.open(f"{DIR}/demo_wrist_rgb.png")),
            seg_ref=np.array(Image.open(f"{DIR}/demo_wrist_seg.png")).astype(bool),
            use_depth=False
        )
        self.DIR = DIR
        self.cartesian_controller = YuMiLeftArmCartesianController()
        self.dof = 3
        self.cap_t = 0.005
        self.cap_r = np.deg2rad(5)
    
    @staticmethod
    def select_best_decomposition(decompositions):
        Rs, ts, normals = decompositions[1], decompositions[2], decompositions[3]
        best_index = -1
        best_score = float('inf')

        for i, (R, t, n) in enumerate(zip(Rs, ts, normals)):
            pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
            roll = np.arctan2(R[2, 1], R[2, 2])

            rotation_score = np.abs(pitch) + np.abs(roll)
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
        pid_rz = PIDController(Kp=0.15, Ki=0.0, Kd=0.05)

        # Initialize error
        error = float('inf')
        start = time.time()
        trajectory = []

        num_iteration = 0
        while num_iteration < 126:

            # Match descriptors
            mkpts_0, mkpts_1, _, _ = self.match_siftlg()
            if len(mkpts_0) <= 3:
                continue

            mkpts_0_norm = normalize_mkpts(mkpts_0, K)
            mkpts_1_norm = normalize_mkpts(mkpts_1, K)

            H_norm, _ = cv2.findHomography(mkpts_0_norm, mkpts_1_norm, cv2.USAC_MAGSAC, 0.001, confidence=0.99999)

            if H_norm is None:
                continue
            dx = H_norm[0, 2]
            dy = H_norm[1, 2]
            # decompositions = cv2.decomposeHomographyMat(H_norm, np.eye(3))
            # best_R, best_t = self.select_best_decomposition(decompositions)
            # print(best_t)
            # # Compute transformation
            # T_delta_cam = np.eye(4)
            # T_delta_cam[:3, :3] = best_R
            # T_delta_cam[:3 ,3] = best_t[0]
            # T_delta_cam[0, 3] = H_norm[0, 2]
            # T_delta_cam[1, 3] = H_norm[1, 2]

            # T_eef_world = yumi.get_curent_T_left()
            # T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)

            # # Update error
            # error = np.linalg.norm(T_delta_cam_inv[:2, 3])
            # print(error)

            # T_delta_eef = d405_T_C_EEF @ T_delta_cam @ pose_inv(d405_T_C_EEF)
            # T_eef_world_new = T_eef_world @ T_delta_eef

            # dx = T_eef_world_new[0, 3] - T_eef_world[0, 3]
            # dy = T_eef_world_new[1, 3] - T_eef_world[1, 3]
            # drz = euler_from_matrix(T_eef_world_new)[-1] - euler_from_matrix(T_eef_world)[-1]

            dx = np.clip(pid_x.update(dx), -self.cap_t, self.cap_t)
            dy = np.clip(pid_y.update(dy), -self.cap_t, self.cap_t)
            # drz = np.clip(pid_rz.update(drz), -self.cap_r, self.cap_r)

            if self.dof == 3:
                current_pose.position.x -= dx
                current_pose.position.y += dy
                # current_rpy[-1] += drz 

            eef_pose = yumi.create_pose_euler(current_pose.position.x, 
                                              current_pose.position.y, 
                                              current_pose.position.z, 
                                              current_rpy[0], 
                                              current_rpy[1], 
                                              current_rpy[2])
            
            self.cartesian_controller.move_eef(eef_pose)
            trajectory.append([current_pose.position.x, current_pose.position.y, np.degrees(current_rpy[-1])])
            num_iteration += 1
        
        rospy.loginfo("VS has aligned or max time allowed passed.")
        np.save("trajectory_hom_implicit.npy", np.array(trajectory))
        visualize_convergence_on_sphere(np.array(trajectory))