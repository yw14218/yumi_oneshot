import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
import yumi_moveit_utils as yumi
from scipy.spatial.transform import Rotation
from trajectory_utils import translation_from_matrix, quaternion_from_matrix, pose_inv, \
                             project3D, create_homogeneous_matrix, apply_transformation_to_waypoints, euler_from_matrix
from ikSolver import IKSolver
import poselib
from poseEstimation import PoseEstimation
import json
import time
from threading import Event, Lock
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF, d415_T_WC as T_WC
import copy
from experiments.scissor.experiment import ScissorExperiment
from experiments.wood.experiment import WoodExperiment
from experiments.pencile_sharpener.experiment import SharpenerExperiment
from lightglue import LightGlue, SuperPoint, match_pair
from xfeat_listener import numpy_image_to_torch
from trajectory_utils import apply_transformation_to_waypoints

def decompose_homography(H, K):
    # Invert the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Compute the B matrix
    B = np.dot(K_inv, np.dot(H, K))

    # Normalize B so that B[2,2] is 1
    B /= B[2, 2]

    # Extract rotation and translation components using SVD
    U, _, Vt = np.linalg.svd(B)
    R = np.dot(U, Vt)

    if np.linalg.det(R) < 0:
        R = -R

    # Extract translation (last column of B, divided by scale factor)
    t = B[:, 2] / np.linalg.norm(B[:, 0])

    # Normalize translation vector
    t /= np.linalg.norm(t)

    return R, t[:2]  # We only need x, y translation

def get_current_stab_3d(T_EEF_World):
    stab_point3d = pose_inv(T_EEF_World @ T_C_EEF) @ T_stab_pose @ T_GRIP_EEF
    # Project the 3D point onto the image plane
    return np.dot(K, stab_point3d[:3, 3])

def move_eef(target):
    ik_left = ik_solver_left.get_ik(target).solution.joint_state.position
    target_joints = ik_left[:7]
    print(f"Number of target joints: {len(target_joints)}")
    msg = JointTrajectory()
    point = JointTrajectoryPoint()
    point.positions = target_joints
    point.time_from_start = rospy.Duration(2)
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = [f"yumi_joint_{i}_l" for i in [1, 2, 7, 3, 4, 5, 6]]
    msg.points.append(point)
    publisher_left.publish(msg)

rospy.init_node('yumi_bayesian_controller', anonymous=True)

publisher_left = rospy.Publisher("/yumi/joint_traj_pos_controller_l/command", JointTrajectory, queue_size=10)
ik_solver_left = IKSolver(group_name="left_arm", ik_link_name="gripper_l_base")

DIR = "experiments/scissor"
OBJ = "black scissor"
T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.136], [0, 0, 0, 1])

# DIR = "experiments/wood"
# OBJ = "wood stand"
# T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.100], [0, 0, 0, 1])

DIR = "experiments/pencile_sharpener"
OBJ = "blue pencile sharpener"
T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.136], [0, 0, 0, 1])

bridge = CvBridge() 
# xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)   

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher
demo_rgb = cv2.imread(f"{DIR}/demo_wrist_rgb.png")[..., ::-1].copy() 
demo_rgb_cuda = numpy_image_to_torch(demo_rgb)

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

pose_estimator = PoseEstimation(
    dir=DIR,
    text_prompt=OBJ,
    visualize=False
)

yumi.init_Moveit()
yumi.reset_init()
# yumi.open_grippers(yumi.LEFT)

user_input = input("Continue? (yes/no): ").lower()
if user_input == "ready":
    pass

yumi.gripper_effort(yumi.LEFT, 20)


yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))
error = 1e6

try:
    while True:
        # Initial head cam alignment
        # diff_xyz, diff_rpy = pose_estimator.decouple_run(output_path=f"{DIR}/", camera_prefix="d415")
        # # rospy.loginfo(f"Diff xyz is {diff_xyz}, diff rpy is {diff_rpy}")
        # bottleneck_left_new = bottleneck_left.copy()
        # bottleneck_left_new[0] += diff_xyz[0]
        # bottleneck_left_new[1] += diff_xyz[1]
        # # bottleneck_left_new[2] += 0.05
        # bottleneck_euler = euler_from_matrix(T_bottleneck_left)
        # bottleneck_euler[-1] += diff_rpy[-1]
        # yumi.plan_left_arm(yumi.create_pose_euler(*bottleneck_left_new[:3], *bottleneck_euler))

        # # Initial wrist cam alignment
        # T_delta_cam = pose_estimator.run(output_path=f"{DIR}/", camera_prefix="d405")
        # T_new_eef_world = yumi.get_curent_T_left() @ T_C_EEF @ T_delta_cam @ pose_inv(T_C_EEF)
        # new_eef_world = translation_from_matrix(T_new_eef_world).tolist() + quaternion_from_matrix(T_new_eef_world).tolist()
        # live_bottleneck_left = project3D(new_eef_world, bottleneck_left)

        # # Move to wrist-cam-depth estimated bottlenck
        # yumi.plan_left_arm(live_bottleneck_left)
        # rospy.sleep(0.1)

        # user_input = input("Continue? (yes/no): ").lower()
        # if user_input == "go":

        # Compute equivalent T_delta_world estimate
        T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)

        # Compute new stab pixel coords and raise to 3d in cam frame
        X = 0
        Y = 0
        for _ in range(30):
            live_rgb_message = rospy.wait_for_message("d405/color/image_rect_raw", Image, timeout=3)
            live_rgb = bridge.imgmsg_to_cv2(live_rgb_message, "rgb8")

            feats0, feats1, matches01 = match_pair(extractor, matcher, demo_rgb_cuda, numpy_image_to_torch(live_rgb))
            matches = matches01['matches']  # indices with shape (K,2)
            mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
            mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)

            H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 5.0)
            x, y = cv2.perspectiveTransform(stab_point_2D_np, H)[0][0]
            err = np.linalg.norm(np.array([x, y]) - np.array(stab_point_2D))
            if err < error:
                X = x
                Y = y
                error = err
                print(f"Current error: {err}, Min error: {error}")

        x, y, z = convert_from_uvd(K, X, Y, stab_3d_cam[-1])

        # Transform to 3d in world frame and transform from gripper pose to eef pose
        x, y, z = (yumi.get_curent_T_left()  @ T_C_EEF @ create_homogeneous_matrix([x, y, z], [0, 0, 0, 1]))[:3, 3]
        T_new_stab_pose = T_delta_world @ T_stab_pose
        new_stab_pose = translation_from_matrix(T_new_stab_pose).tolist() + quaternion_from_matrix(T_new_stab_pose).tolist()
        
        # Step 3: Call the decomposition function
        R, t_xy = decompose_homography(H, K)
        # Output the results
        # print("Translation (x, y):", t_xy)
        # print("Rotation Matrix:\n", np.degrees(np.arctan2(H[1, 0], H[0, 0])))

        # Romove roll and pitch componenents
        new_stab_pose = project3D(new_stab_pose, stab_pose)
        T_new_stab_pose_refined = create_homogeneous_matrix([x, y, z], new_stab_pose[3:]) @ pose_inv(T_GRIP_EEF)

        # Re-estimate T_delta_world using updated live_stab_x, live_stab_y of eef       
        T_new_stab_pose_refined[2, 3] = stab_pose[-1]
        
        T_delta_world =  T_new_stab_pose_refined @ pose_inv(T_stab_pose)

        # # # Apply T_delta_world to rest of trajectories
        live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)

        # move_eef(yumi.create_pose(*live_waypoints[0]))
        rospy.sleep(0.1)
        SharpenerExperiment.replay(live_waypoints)
        # yumi.plan_left_arm(live_waypoints[0])
        
        # user_input = input("Continue? (yes/no): ").lower()
        # if user_input == "yes":
        #     SharpenerExperiment.reset()
        #     user_input = input("Continue? (yes/no): ").lower()
        #     if user_input == "ready":
        #         pass
        # elif user_input == "reset":
        #     SharpenerExperiment.reset()
        #     break
        # else:
        #     break
    rospy.spin()

except Exception as e:
    print(f"Error: {e}")