import numpy as np
import rospy
import ros_numpy
import time
import moveit_utils.yumi_moveit_utils as yumi
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from camera_utils import solve_transform_3d, d405_K as K, d405_T_C_EEF
from trajectory_utils import pose_inv, euler_from_quat, euler_from_matrix
from moveit_utils.cartesian_control import YuMiCartesianController
from lightglue import SIFT, LightGlue
from lightglue.utils import load_image, rbd
from vis import visualize_convergence_on_sphere
import warnings
import matplotlib.pyplot as plt
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
    
class RelCamVS:
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
    
    def match_siftlg(self, live_image_dir, filter_seg=True):
        live_image = load_image(live_image_dir).cuda()
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

        pid_x = PIDController(Kp=0.15, Ki=0.0, Kd=0.05)
        pid_y = PIDController(Kp=0.15, Ki=0.0, Kd=0.05)
        pid_rz = PIDController(Kp=0.15, Ki=0.0, Kd=0.05)

        # Initialize error
        error = float('inf')
        start = time.time()
        trajectory = []

        num_iteration = 0
        while error > 0.005 and num_iteration < 50:
            
            # Save RGB and depth images
            live_rgb_dir, live_depth_dir = self.save_rgbd()
            
            # Match descriptors
            mkpts_0, mkpts_1 = self.match_siftlg(live_rgb_dir)

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