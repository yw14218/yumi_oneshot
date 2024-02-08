#!/usr/bin/env python3

import rospy
import yumi_moveit_utils as yumi
import json
from get_fk import GetFK
import numpy as np
from scipy.spatial.transform import Rotation as R
from replay_trajectory import dict_to_joint_state
from trajectory_utils import apply_transformation_to_waypoints
from replay_trajectory import dict_to_joint_state, move_upwards
from poseEstimation import PoseEstimation
from collect_left_arm_demo import dict_to_ros_pose

def load_grasp_pose():

    gfk_left = GetFK('gripper_l_base', 'world')
    file_name="data/split_lego/left.json"

    with open(file_name) as f:
        joint_states = json.load(f)

    msg = dict_to_joint_state(joint_states[0])

    return gfk_left.get_fk(msg).pose_stamped[0].pose 

def load_left_demo_poses():
    file_name="data/lego_split/left_demo_poses.json"

    with open(file_name) as f:
        poses = json.load(f)
    return dict_to_ros_pose(poses)

def run():

    rospy.init_node('yumi_moveit_demo')
    yumi.init_Moveit()

    bottle_neck_poses = load_left_demo_poses()
    bottle_neck_pose = load_grasp_pose()


    pose_estimator = PoseEstimation(
        text_prompt="lego",
        demo_rgb_path="data/lego_split/demo_rgb.png",
        demo_depth_path="data/lego_split/demo_depth.png",
        demo_mask_path="data/lego_split/demo_mask.png",
        intrinsics_path="handeye/intrinsics.npy",
        T_WC_path="handeye/T_WC_head.npy"
    )

    while True:
        yumi.reset_init()

        T_delta_world = pose_estimator.run(output_path="data/lego_split/")

        # Planning and executing transferred trajectories
        waypoints = [bottle_neck_pose]
        waypoints_np = np.array([[waypoint.position.x, waypoint.position.y, waypoint.position.z,
                                waypoint.orientation.x, waypoint.orientation.y, waypoint.orientation.z,
                                waypoint.orientation.w] for waypoint in waypoints])

        transformed_fine_waypoints = apply_transformation_to_waypoints(waypoints_np, T_delta_world)
        transferred_waypoints = [yumi.create_pose(*waypoint) for waypoint in transformed_fine_waypoints]
            
        yumi.group_l.set_pose_target(transferred_waypoints[0])
        plan = yumi.group_l.plan()
        yumi.group_l.go(wait=True)

        rospy.sleep(2)

        # Additional movement planning
        yumi.gripper_effort(yumi.LEFT, 20)

        yumi.group_l.set_pose_target(bottle_neck_poses["end"])
        plan = yumi.group_l.plan()
        yumi.group_l.go(wait=True)


        # User input to continue or break the loop
        user_input = input("Continue? (yes/no): ").lower()
        if user_input != 'yes':
            break

    rospy.spin()




if __name__ == '__main__':
    try:
       run()
    except Exception as e:
        print(f"Error: {e}")