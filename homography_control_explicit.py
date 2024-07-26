import rospy
import torch
import cv2
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction, SequentialDomainReductionTransformer
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation
from trajectory_utils import *
from moveit_utils.ikSolver import IKSolver
import poselib
from poseEstimation import PoseEstimation
import json
import time
from threading import Event, Lock
from camera_utils import d405_K as K, d405_T_C_EEF as T_C_EEF
from matplotlib import pyplot as plt
from matplotlib import gridspec
from lightglue import LightGlue, SuperPoint, SIFT, match_pair
from geometry_msgs.msg import geometry_msgs
import moveit_utils.yumi_moveit_utils as yumi
import matplotlib.pyplot as plt
from vis import *
from collections import deque

class ImageListener:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.live_rgb_batch = []
        self.bridge = CvBridge() 
        self.image_event = Event()
        self.lock = Lock()
        self.subscriber = rospy.Subscriber("d405/color/image_rect_raw", Image, self.image_callback)
        self.homography_buffer = deque(maxlen=3)

    def image_callback(self, msg):
        with self.lock:
            if len(self.live_rgb_batch) < self.batch_size:
                self.live_rgb_batch.append(self.bridge.imgmsg_to_cv2(msg, "rgb8"))
                if len(self.live_rgb_batch) == self.batch_size:
                    self.image_event.set()

    @staticmethod
    def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
        """Normalize the image tensor and reorder the dimensions."""
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        elif image.ndim == 2:
            image = image[None]  # add channel axis
        else:
            raise ValueError(f"Not an image: {image.shape}")
        return torch.tensor(image / 255.0, dtype=torch.float, device="cuda")

    @staticmethod
    def normalize_points(points, K):
        K_inv = np.linalg.inv(K)
        return (K_inv @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :2]

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

        return Rotation.from_matrix(best_R).as_euler('xyz'), best_t

    def observe(self):
        with self.lock:
            self.live_rgb_batch = []
        self.image_event.clear()

        # Wait for batch_size images to be collected or timeout
        collected = self.image_event.wait(1)

        if not collected:
            rospy.logwarn("Timeout occurred while waiting for images.")
            raise NotImplementedError

        with self.lock:
            dx = 0
            dy = 0
            drz = 0
            dz = 0
            for live_rgb in self.live_rgb_batch:
                feats0, feats1, matches01 = match_pair(extractor, matcher, demo_rgb_cuda, self.numpy_image_to_torch(live_rgb))
                matches = matches01['matches']  # indices with shape (K,2)
                mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
                mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)

                # Normalize your points
                mkpts_0_norm = self.normalize_points(mkpts_0, K)
                mkpts_1_norm = self.normalize_points(mkpts_1, K)

                try:
                    H_norm, _ = cv2.findHomography(mkpts_0_norm, mkpts_1_norm, cv2.USAC_MAGSAC, 0.001, confidence=0.99999)
                    if H_norm is not None:
                        H = K @ H_norm @ np.linalg.inv(K)
                        dx, dy = cv2.perspectiveTransform(stab_point_2D_np, H)[0][0]
                        decompositions = cv2.decomposeHomographyMat(H_norm, np.eye(3))
                        best_R, best_t = self.select_best_decomposition(decompositions)
                        drz = best_R[-1]
                        # Compute the determinant of H
                        det_H = np.linalg.det(H)
                        # Compute the ratio Z/Z*
                        # Note: Z/Z* = (det(H))^(-1/3)
                        Z_ratio = det_H ** (-1/3)
                        # Compute the error using natural logarithm
                        dz = np.log(Z_ratio)
                except cv2.error:
                    pass

            return (dx, dy), drz, dz

def get_current_stab_3d(T_EEF_World):
    stab_point3d = pose_inv(T_EEF_World @ T_C_EEF) @ T_stab_pose @ T_GRIP_EEF
    # Project the 3D point onto the image plane
    return np.dot(K, stab_point3d[:3, 3])

def move_eef(new_eef_pose):

    ik_left = ik_solver_left.get_ik(new_eef_pose).solution.joint_state.position
    target_joints = ik_left[:7]
    
    msg = JointTrajectory()
    point = JointTrajectoryPoint()
    point.positions = target_joints
    point.time_from_start = rospy.Duration(0.05)
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = [f"yumi_joint_{i}_l" for i in [1, 2, 7, 3, 4, 5, 6]]
    msg.points.append(point)
    publisher_left.publish(msg)

rospy.init_node('yumi_bayesian_controller', anonymous=True)

publisher_left = rospy.Publisher("/yumi/joint_traj_pos_controller_l/command", JointTrajectory, queue_size=10)
ik_solver_left = IKSolver(group_name="left_arm", ik_link_name="gripper_l_base")
imageListener = ImageListener(batch_size=1)

DIR = "experiments/pencil_sharpener"
OBJ = "blue pencile sharpener"

DIR = "experiments/scissor"
OBJ = "black scissor"


# DIR = "experiments/wood"
# OBJ = "wooden stand"
from experiments.wood.experiment import WoodExperiment as Experiment

T_GRIP_EEF = create_homogeneous_matrix([-0.07, 0.08, 0.2], [0, 0, 0, 1])

pose_estimator = PoseEstimation(
    dir=DIR,
    text_prompt=OBJ,
    visualize=False
)

file_name = f"{DIR}/demo_bottlenecks.json"
with open(file_name) as f:
    dbn = json.load(f)
demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

bottleneck_left = demo_waypoints[0].tolist()
bottleneck_right = demo_waypoints[1].tolist()
stab_pose = dbn["bottleneck_left"]
T_bottleneck_left = create_homogeneous_matrix(bottleneck_left[:3], bottleneck_left[3:])
T_stab_pose = create_homogeneous_matrix(stab_pose[:3], stab_pose[3:])
stab_3d_cam = get_current_stab_3d(T_EEF_World=T_bottleneck_left)

# Normalize the coordinates to get the 2D image point
stab_point_2D = stab_3d_cam[:2] / stab_3d_cam[2]
stab_point_2D_np = np.array(stab_point_2D, dtype=np.float32).reshape(-1, 1, 2)
# xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)   

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher
# extractor = SIFT(max_num_keypoints=1024, backend='pycolmap').eval().cuda()
# matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher
demo_rgb = cv2.imread(f"{DIR}/demo_wrist_rgb.png")[..., ::-1].copy() 
demo_seg = cv2.imread(f"{DIR}/demo_wrist_seg.png")
demo_rgb_cuda = imageListener.numpy_image_to_torch(demo_rgb * demo_seg.astype(bool))

yumi.init_Moveit()

# yumi.reset_init()
# bottleneck_left[2] += 0.05
yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))
user_input = input("Continue? (yes/no): ").lower()
if user_input == "ready":
    pass

# yumi.gripper_effort(yumi.LEFT, 20)

# diff_xyz, diff_rpy = pose_estimator.decouple_run(output_path=f"{DIR}/", camera_prefix="d415")
# # rospy.loginfo(f"Diff xyz is {diff_xyz}, diff rpy is {diff_rpy}")
# bottleneck_left_new = bottleneck_left.copy()
# bottleneck_left_new[0] += diff_xyz[0]
# bottleneck_left_new[1] += diff_xyz[1]
# # bottleneck_left_new[2] += 0.05
# bottleneck_euler = euler_from_matrix(T_bottleneck_left)
# bottleneck_euler[-1] += diff_rpy[-1]
# yumi.plan_left_arm(yumi.create_pose_euler(*bottleneck_left_new[:3], *bottleneck_euler))
yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))


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
    
def iterative_learning_control(demo_pixel, K, stab_3d_cam, max_iterations=150, threshold=0.1):
    current_pose = yumi.get_current_pose(yumi.LEFT).pose
    current_rpy = euler_from_quat([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
    control_input_x = 0
    control_input_y = 0
    errors = []
    pid_x = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
    pid_y = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
    pid_theta = PIDController(Kp=0.1, Ki=0.0, Kd=0.02)
    pid_z = PIDController(Kp=0.5, Ki=0.0, Kd=0.1)
    trajectory = []

    for iteration in range(max_iterations):
        # Capture current image and detect projection pixel
        current_pixel, delta_rz, delta_z = imageListener.observe()
        
        print(delta_z)
        # Calculate pixel error
        delta_x = demo_pixel[0] - current_pixel[0]
        delta_y = demo_pixel[1] - current_pixel[1]
        error = np.linalg.norm(np.array([demo_pixel]) - np.array(current_pixel), ord=1)
        
        # Check if error is within threshold
        if (abs(delta_x) < threshold and abs(delta_y) < threshold and error < 1) or iteration == max_iterations - 1:
            print(abs(delta_x), abs(delta_y))
            # T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
            # live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
            # Experiment.replay(live_waypoints)
            break

        delta_X = delta_x * stab_3d_cam[-1] / K[0][0]
        delta_Y = delta_y * stab_3d_cam[-1] / K[1][1]

        errors.append(error)

        control_input_x = pid_x.update(delta_X)
        control_input_y = pid_y.update(delta_Y)
        control_input_z_rot = pid_theta.update(delta_rz)
        # control_input_z = pid_z.update(delta_z)
        

        # if abs(control_input_x) >= 0.002:
        #     control_input_x = 0.002 if control_input_x > 0 else -0.002
        # if abs(control_input_y) >= 0.002:
        #     control_input_y = 0.002 if control_input_y > 0 else -0.002
        # if abs(control_input_z_rot) >= 0.02:
        #     control_input_z_rot = 0.02 if control_input_z_rot > 0 else -0.02
        # if abs(control_input_z) >= 0.001:
        #     control_input_z = 0.001 if control_input_z > 0 else -0.001

        rospy.loginfo(f"Step {iteration + 1}, Error is : {error:.4g}, delta_x: {control_input_x:.4g}, delta_y: {control_input_y:.4g}")
    
        # Move robot by the updated control input
        current_pose.position.x += control_input_x
        current_pose.position.y -= control_input_y
        # current_pose.position.z -= control_input_z
        current_rpy[-1] -= control_input_z_rot
        
        trajectory.append([current_pose.position.x, current_pose.position.y, np.degrees(current_rpy[-1])])

        new_pose = yumi.create_pose_euler(current_pose.position.x, current_pose.position.y, current_pose.position.z, current_rpy[0], current_rpy[1], current_rpy[2])

        move_eef(new_pose)

    # np.save("trajectory_hom_no_depth.npy", np.array(trajectory))
    visualize_convergence_on_sphere(np.array(trajectory))
    
    return current_pose

iterative_learning_control(demo_pixel=stab_point_2D, K=K, stab_3d_cam=stab_3d_cam)

rospy.spin()