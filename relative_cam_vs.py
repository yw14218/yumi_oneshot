import numpy as np
import rospy
import ros_numpy
import time
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from camera_utils import solve_transform_3d, normalize_mkpts, d405_K as K, d405_T_C_EEF
from trajectory_utils import pose_inv, euler_from_quat, euler_from_matrix
from moveit_utils.cartesian_control import YuMiLeftArmCartesianController
from lightglue import SIFT, LightGlue
from lightglue.utils import load_image, rbd
from vis import visualize_convergence_on_sphere
import warnings
import matplotlib.pyplot as plt
from base_servoer import PIDController, SiftLightGlueVisualServoer
warnings.filterwarnings("ignore")

    
class RelCamVS(SiftLightGlueVisualServoer):
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

        self.cap_t = 0.005
        self.cap_r = np.deg2rad(5)
    
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
        while error > 0.001 and num_iteration < 126:
            
            # Match descriptors
            mkpts_0, mkpts_1, depth_cur, highest_confidence_index = self.match_siftlg(filter_seg=False)
            if mkpts_0 is None or len(mkpts_0) <= 3:
                continue
            
            # Compute mean dx and dy
            dx, dy = mkpts_1[highest_confidence_index] - mkpts_0[highest_confidence_index]
            dx /= K[0][0]
            dy /= K[1][1]

            # print(dx, dy)
            # Compute transformation
            T_delta_cam = solve_transform_3d(mkpts_0, mkpts_1, self.depth_ref, depth_cur, K)
            T_eef_world = yumi.get_curent_T_left()
            T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)

             # Update error
            error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            print(error)

            T_delta_eef = d405_T_C_EEF @ T_delta_cam @ pose_inv(d405_T_C_EEF)
            T_eef_world_new = T_eef_world @ T_delta_eef

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
        
        rospy.loginfo("DINO has aligned or max time allowed passed.")
        np.save("trajectory_kps_highest_confidence", np.array(trajectory))
        visualize_convergence_on_sphere(np.array(trajectory))