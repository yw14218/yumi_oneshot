import rospy
import torch
import cv2
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction, SequentialDomainReductionTransformer
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation
from trajectory_utils import translation_from_matrix, quaternion_from_matrix, pose_inv, project3D, create_homogeneous_matrix, apply_transformation_to_waypoints
from ikSolver import IKSolver
import poselib
from poseEstimation import PoseEstimation
import json
import time
from threading import Event, Lock
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF
from matplotlib import pyplot as plt
from matplotlib import gridspec
from lightglue import LightGlue, SuperPoint, match_pair
from xfeat_listener import numpy_image_to_torch, decompose_homography
from geometry_msgs.msg import geometry_msgs
import yumi_moveit_utils as yumi
from experiments.pencile_sharpener.experiment import SharpenerExperiment


def create_pose(x_p, y_p, z_p, x_o, y_o, z_o, w_o):
    """Creates a pose using quaternions

    Creates a pose for use with MoveIt! using XYZ coordinates and XYZW
    quaternion values

    :param x_p: The X-coordinate for the pose
    :param y_p: The Y-coordinate for the pose
    :param z_p: The Z-coordinate for the pose
    :param x_o: The X-value for the orientation
    :param y_o: The Y-value for the orientation
    :param z_o: The Z-value for the orientation
    :param w_o: The W-value for the orientation
    :type x_p: float
    :type y_p: float
    :type z_p: float
    :type x_o: float
    :type y_o: float
    :type z_o: float
    :type w_o: float
    :returns: Pose
    :rtype: PoseStamped
    """
    pose_target = geometry_msgs.msg.Pose()
    pose_target.position.x = x_p
    pose_target.position.y = y_p
    pose_target.position.z = z_p
    pose_target.orientation.x = x_o
    pose_target.orientation.y = y_o
    pose_target.orientation.z = z_o
    pose_target.orientation.w = w_o
    return pose_target

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
    point.time_from_start = rospy.Duration(0.1)
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = [f"yumi_joint_{i}_l" for i in [1, 2, 7, 3, 4, 5, 6]]
    msg.points.append(point)
    publisher_left.publish(msg)

rospy.init_node('yumi_bayesian_controller', anonymous=True)

publisher_left = rospy.Publisher("/yumi/joint_traj_pos_controller_l/command", JointTrajectory, queue_size=10)
ik_solver_left = IKSolver(group_name="left_arm", ik_link_name="gripper_l_base")


DIR = "experiments/pencile_sharpener"
OBJ = "blue pencile sharpener"
T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.136], [0, 0, 0, 1])

file_name = f"{DIR}/demo_bottlenecks.json"
with open(file_name) as f:
    dbn = json.load(f)
demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

bottleneck_left = demo_waypoints[0].tolist()
bottleneck_right = demo_waypoints[1].tolist()
stab_pose = dbn["grasp_right"]
T_bottleneck_left = create_homogeneous_matrix(bottleneck_left[:3], bottleneck_left[3:])
T_stab_pose = create_homogeneous_matrix(stab_pose[:3], stab_pose[3:])
stab_3d_cam = get_current_stab_3d(T_EEF_World=T_bottleneck_left)

# Normalize the coordinates to get the 2D image point
stab_point_2D = stab_3d_cam[:2] / stab_3d_cam[2]
stab_point_2D_np = np.array(stab_point_2D, dtype=np.float32).reshape(-1, 1, 2)

bridge = CvBridge() 
# xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)   

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher
demo_rgb = cv2.imread(f"{DIR}/demo_wrist_rgb.png")[..., ::-1].copy() 
demo_rgb_cuda = numpy_image_to_torch(demo_rgb)

yumi.init_Moveit()

def get_error():
    error = 1e6
    for _ in range(30):
        live_rgb_message = rospy.wait_for_message("d405/color/image_rect_raw", Image, timeout=3)
        live_rgb = bridge.imgmsg_to_cv2(live_rgb_message, "rgb8")

        feats0, feats1, matches01 = match_pair(extractor, matcher, demo_rgb_cuda, numpy_image_to_torch(live_rgb))
        matches = matches01['matches']  # indices with shape (K,2)
        mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
        mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)

        H, _ = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 5.0)
        x, y = cv2.perspectiveTransform(stab_point_2D_np, H)[0][0]
        err = np.linalg.norm(np.array([x, y]) - np.array(stab_point_2D))
        if error < err:
            error = err
    return err

yumi.reset_init()
move_eef(create_pose(*bottleneck_left))
# move_eef(create_pose(*bottleneck_right))

user_input = input("Continue? (yes/no): ").lower()
if user_input == "ready":
    pass

yumi.gripper_effort(yumi.LEFT, 20)
# yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))

# global_error = get_error()

# x = 0
# y = 0

# for i in range(-5, 5, 1):
#     new_pose = bottleneck_left.copy()
#     new_pose[0] += i * 0.001
#     move_eef(create_pose(*new_pose))
#     err = get_error()
#     if err < global_error:
#         global_error = err
#         x = i
        
# move_eef(create_pose(*bottleneck_left))

# for j in range(-5, 5, 1):
#     new_pose = bottleneck_left.copy()
#     new_pose[1] += i * 0.001
#     move_eef(create_pose(*new_pose))
#     err = get_error()
#     if err < global_error:
#         global_error = err
#         y = i

# print("Min error found: ", x, y)

def iterative_learning_control(demo_pixel, K, stab_3d_cam, max_iterations=60, gain=0.1, threshold=1):
    current_pose = yumi.get_current_pose(yumi.LEFT)
    control_input_x = 0
    control_input_y = 0

    for iteration in range(max_iterations):
        # Capture current image and detect projection pixel
        current_pixel = detect_projection_pixel()
        
        # Calculate pixel error
        delta_x = demo_pixel[0] - current_pixel[0]
        delta_y = demo_pixel[1] - current_pixel[1]
        
        # Check if error is within threshold
        if abs(delta_x) < threshold and abs(delta_y) < threshold:
            T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
            live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
            SharpenerExperiment.replay(live_waypoints)
            break

        delta_X = delta_x * stab_3d_cam[-1] / K[0][0]
        delta_Y = delta_y * stab_3d_cam[-1] / K[1][1]

        error = np.linalg.norm(np.array([demo_pixel]) - np.array(current_pixel))

        print(delta_X, delta_Y, error)
        control_input_x = int(delta_X * 1000) * gain
        control_input_y = int(delta_Y * 1000) * gain
        
        # Move robot by the updated control input
        current_pose.pose.position.x += control_input_x * 0.001
        current_pose.pose.position.y -= control_input_y * 0.001
        move_eef(current_pose.pose)

    T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
    live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
    SharpenerExperiment.replay(live_waypoints)

    return current_pose

def detect_projection_pixel():
    X = 0
    Y = 0
    for _ in range(5):
        live_rgb_message = rospy.wait_for_message("d405/color/image_rect_raw", Image, timeout=3)
        live_rgb = bridge.imgmsg_to_cv2(live_rgb_message, "rgb8")

        feats0, feats1, matches01 = match_pair(extractor, matcher, demo_rgb_cuda, numpy_image_to_torch(live_rgb))
        matches = matches01['matches']  # indices with shape (K,2)
        mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
        mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)

        H, _ = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 5.0)
        x, y = cv2.perspectiveTransform(stab_point_2D_np, H)[0][0]
        X += x
        Y += y

    return X/5, Y/5

iterative_learning_control(demo_pixel=stab_point_2D, K=K, stab_3d_cam=stab_3d_cam)

rospy.spin()
