import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yumi_moveit_utils as yumi
from scipy.spatial.transform import Rotation
from trajectory_utils import translation_from_matrix, quaternion_from_matrix, pose_inv, \
                             project3D, create_homogeneous_matrix, apply_transformation_to_waypoints, euler_from_matrix
from ikSolver import IKSolver
import poselib
from poseEstimation import PoseEstimation
import json
import time
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF, d415_T_WC as T_WC
import copy
from experiments.scissor.experiment import ScissorExperiment
from experiments.wood.experiment import WoodExperiment
from experiments.pencile_sharpener.experiment import SharpenerExperiment
from trajectory_utils import apply_transformation_to_waypoints


rospy.init_node('yumi_dense_ICP', anonymous=True)


# DIR = "experiments/scissor"
# OBJ = "black scissor"

DIR = "experiments/wood"
OBJ = "wood stand"

# DIR = "experiments/pencile_sharpener"
# OBJ = "blue pencile sharpener"


demo_rgb = cv2.imread(f"{DIR}/demo_wrist_rgb.png")[..., ::-1].copy() 

file_name = f"{DIR}/demo_bottlenecks.json"
with open(file_name) as f:
    dbn = json.load(f)
demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

bottleneck_left = demo_waypoints[0].tolist()
bottleneck_right = demo_waypoints[1].tolist()

# Normalize the coordinates to get the 2D image point

pose_estimator = PoseEstimation(
    dir=DIR,
    text_prompt=OBJ,
    visualize=True
)

yumi.init_Moveit()
yumi.reset_init()

yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))

user_input = input("Ready: ").lower()
if user_input == "ready":
    pass

try:
    while True:
        # Initial head cam alignment
        T_delta_cam = pose_estimator.run(output_path=f"{DIR}/", camera_prefix="d415")
        T_WC = np.load(pose_estimator.T_WC_path)
        T_delta_world = T_WC @ T_delta_cam @ pose_inv(T_WC)
        # T_delta_world = pose_estimator.run_image_match(output_path=f"{self.dir}/", camera_prefix="d415")
        rospy.loginfo("T_delta_world is {0}".format(T_delta_world))
        live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=False)
        yumi.plan_both_arms(live_waypoints[0], live_waypoints[1])

        # Apply T_delta_world to rest of trajectories
        live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=False)
        WoodExperiment.replay(live_waypoints)
        
        user_input = input("Continue? (yes/no): ").lower()
        if user_input == "yes":
            WoodExperiment.reset()
            user_input = input("Ready? (yes/no): ").lower()
            if user_input == "ready":
                pass
        elif user_input == "reset":
            WoodExperiment.reset()
            break
        else:
            break
    rospy.spin()

except Exception as e:
    print(f"Error: {e}")

# 2 10
# 0 10
# 0 6