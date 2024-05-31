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
                             project3D, create_homogeneous_matrix, apply_transformation_to_waypoints
from ikSolver import IKSolver
import poselib
from poseEstimation import PoseEstimation
import json
import time
from threading import Event, Lock
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF
import copy
from experiments.scissor.experiment import ScissorExperiment

def get_current_stab_3d(T_EEF_World):
    stab_point3d = pose_inv(T_EEF_World @ T_C_EEF) @ T_stab_pose @ T_GRIP_EEF
    # Project the 3D point onto the image plane
    return np.dot(K, stab_point3d[:3, 3])

rospy.init_node('yumi_bayesian_controller', anonymous=True)
DIR = "experiments/scissor"
OBJ = "scissor"
bridge = CvBridge() 
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)   
demo_rgb = cv2.imread(f"{DIR}/demo_wrist_rgb.png")[..., ::-1].copy() 


file_name = f"{DIR}/demo_bottlenecks.json"
with open(file_name) as f:
    dbn = json.load(f)
demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

bottleneck_left = demo_waypoints[0].tolist()
bottleneck_right = demo_waypoints[1].tolist()
stab_pose = demo_waypoints[3].tolist()
T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.136], [0, 0, 0, 1])
T_bottleneck_left = create_homogeneous_matrix(bottleneck_left[:3], bottleneck_left[3:])
T_stab_pose = create_homogeneous_matrix(stab_pose[:3], stab_pose[3:])
stab_3d_cam = get_current_stab_3d(T_EEF_World=T_bottleneck_left)
# Normalize the coordinates to get the 2D image point
stab_point_2D = stab_3d_cam[:2] / stab_3d_cam[2]
stab_point_2D_np = np.array(stab_point_2D, dtype=np.float32).reshape(-1, 1, 2)

yumi.init_Moveit()
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
        new_eef_world = translation_from_matrix(T_new_eef_world).tolist() + quaternion_from_matrix(T_new_eef_world).tolist()
        live_bottleneck_left = project3D(new_eef_world, bottleneck_left)

        # Move to wrist-cam-depth estimated bottlenck
        yumi.plan_left_arm(live_bottleneck_left)
        rospy.sleep(1)

        # Compute equivalent T_delta_world estimate
        T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)

        # Compute new stab pixel coords and raise to 3d in cam frame
        live_rgb_message = rospy.wait_for_message("d405/color/image_rect_raw", Image, timeout=3)
        live_rgb = bridge.imgmsg_to_cv2(live_rgb_message, "rgb8")
        mkpts_0, mkpts_1 = xfeat.match_xfeat_star(demo_rgb, live_rgb, top_k = 4096)
        H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 5.0)
        x, y = cv2.perspectiveTransform(stab_point_2D_np, H)[0][0]
        x, y, z = convert_from_uvd(K, x, y, stab_3d_cam[-1]/1000)

        # Transform to 3d in world frame and transform from gripper pose to eef pose
        x, y, z = (yumi.get_curent_T_left()  @ T_C_EEF @ create_homogeneous_matrix([x, y, z], [0, 0, 0, 1]))[:3, 3]
        T_live_stab_pose = project3D(T_delta_world @ T_stab_pose, T_stab_pose)
        T_eef_world  = create_homogeneous_matrix([x, y, z], quaternion_from_matrix(T_live_stab_pose)) @ pose_inv(T_GRIP_EEF)
        
        # Re-estimate T_delta_world using updated live_stab_x, live_stab_y of eef
        x, y, z = T_eef_world[:3, 3].tolist()        
        T_live_stab_pose[0, 3] = x
        T_live_stab_pose[1, 3] = y
        T_live_stab_pose[2, 3] = stab_pose[-1]
        T_delta_world =  T_live_stab_pose @ pose_inv(T_stab_pose)

        # Apply T_delta_world to rest of trajectories
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