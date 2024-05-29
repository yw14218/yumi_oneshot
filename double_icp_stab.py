import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
import yumi_moveit_utils as yumi
from scipy.spatial.transform import Rotation
from trajectory_utils import translation_from_matrix, quaternion_from_matrix, pose_inv, project3D, create_homogeneous_matrix, apply_transformation_to_waypoints
from ikSolver import IKSolver
import poselib
from poseEstimation import PoseEstimation
import json
import time
from threading import Event, Lock
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF
import copy
from experiments.scissor.experiment import ScissorExperiment

rospy.init_node('yumi_bayesian_controller', anonymous=True)
DIR = "experiments/scissor"
OBJ = "scissor"
key_point_demo = np.array([499.86090324, 103.8931002], dtype=np.float32).reshape(-1, 1, 2)
demo_image_path = f"{DIR}/demo_wrist_rgb.png"
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)   
demo_rgb = cv2.imread(demo_image_path)[..., ::-1].copy() 
bridge = CvBridge() 
yumi.init_Moveit()

file_name = f"{DIR}/demo_bottlenecks.json"
with open(file_name) as f:
    dbn = json.load(f)
demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

bottleneck_left = demo_waypoints[0].tolist()
bottleneck_right = demo_waypoints[1].tolist()
yumi.reset_init(yumi.RIGHT)
yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))
yumi.open_grippers(yumi.LEFT)

try:
    pose_estimator = PoseEstimation(
        dir=DIR,
        text_prompt=OBJ,
        visualize=False
    )

    while True:
        # Initial wrist cam alignment
        T_delta_cam = pose_estimator.run(output_path=f"{DIR}/", camera_prefix="d405")
        T_new_eef_world = yumi.get_curent_T_left() @ T_C_EEF @ T_delta_cam @ pose_inv(T_C_EEF)
        xyz = translation_from_matrix(T_new_eef_world).tolist()
        quaternion = quaternion_from_matrix(T_new_eef_world).tolist()
        pose_new_eef_world_l = project3D(xyz + quaternion, demo_waypoints[0])
        yumi.plan_left_arm(pose_new_eef_world_l)
        rospy.sleep(1)

        stablizing_pose = demo_waypoints[3]
        T_stab_pose = create_homogeneous_matrix(stablizing_pose[:3], stablizing_pose[3:])
        T_bottleneck_left = create_homogeneous_matrix(bottleneck_left[:3], bottleneck_left[3:])
        T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)

        depth_message = rospy.wait_for_message("d405/aligned_depth_to_color/image_raw", Image, timeout=5)
        live_rgb_message = rospy.wait_for_message("d405/color/image_rect_raw", Image, timeout=5)
        depth_image = bridge.imgmsg_to_cv2(depth_message, "32FC1")
        live_rgb = bridge.imgmsg_to_cv2(live_rgb_message, "rgb8")
        mkpts_0, mkpts_1 = xfeat.match_xfeat_star(demo_rgb, live_rgb, top_k = 4096)
        H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 5.0)
        key_point_live_hom = cv2.perspectiveTransform(key_point_demo, H)
        x, y = key_point_live_hom[0][0]
        z = np.array(depth_image)[int(y), int(x)]
        x, y, z = convert_from_uvd(K, x, y, z /1000)
        T_EEF_WORLD = yumi.get_curent_T_left() 
        xyz_world = (T_EEF_WORLD @ T_C_EEF @ create_homogeneous_matrix([x, y, z], [0, 0, 0, 1]))[:3, 3]
        x, y, z = xyz_world

        T_live_stab_pose = T_delta_world @ T_stab_pose
        T_grip_world = create_homogeneous_matrix([x, y , z], quaternion_from_matrix(T_live_stab_pose))
        T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.136], [0, 0, 0, 1])
        T_eef_world = T_grip_world @ pose_inv(T_GRIP_EEF)

        xyz_eef_world = T_eef_world[:3, 3].tolist()
        x, y, z = xyz_eef_world
        
        T_live_stab_pose[0, 3] = x
        T_live_stab_pose[1, 3] = y
        T_live_stab_pose[2, 3] = stablizing_pose[2]
        T_delta_world =  T_live_stab_pose @ pose_inv(T_stab_pose)
        live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
        ScissorExperiment.replay(live_waypoints, yumi.RIGHT)
        
        user_input = input("Continue? (yes/no): ").lower()
        if user_input != 'yes':
            break
        else:
            yumi.open_grippers(yumi.LEFT)
            yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))
            yumi.reset_init(yumi.RIGHT)
            rospy.sleep(1)

    rospy.spin()

except Exception as e:
    print(f"Error: {e}")