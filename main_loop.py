#!/usr/bin/env python3

import rospy
import copy
import json
import argparse
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import yumi_moveit_utils as yumi
import pytransform3d.trajectories as ptr
from get_fk import GetFK
from sensor_msgs.msg import JointState
from trajectory_utils import filter_joint_states, apply_transformation_to_waypoints
from scipy.spatial.transform import Rotation as R
from replay_trajectory import dict_to_joint_state, display_trajectory, move_upwards
from poseEstimation import PoseEstimation

def run(gfk_left):
    try:
        file_name="lift_lego_left.json"

        with open(file_name) as f:
            joint_states = json.load(f)

        filtered_joint_states = filter_joint_states(joint_states, 0.01)
        msgs = [dict_to_joint_state(filtered_joint_state) for filtered_joint_state in filtered_joint_states]
        rospy.loginfo("{} waypoints in the trajectory".format(len(msgs)))

        eef_poses_left = [gfk_left.get_fk(msg) for msg in msgs]
        assert len(eef_poses_left) == len(msgs), "Error in computing FK"

        pose_estimator = PoseEstimation(
            text_prompt="lego",
            demo_rgb_path="data/lego/demo_rgb.png",
            demo_depth_path="data/lego/demo_depth.png",
            demo_mask_path="data/lego/demo_mask.png",
            intrinsics_path="handeye/intrinsics.npy",
            T_WC_path="handeye/T_WC_head.npy"
        )

        while True:

            T_delta_world = pose_estimator.run()

            # Planning and executing transferred trajectories
            waypoints = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses_left]
            waypoints_np = np.array([[waypoint.position.x, waypoint.position.y, waypoint.position.z,
                                    waypoint.orientation.x, waypoint.orientation.y, waypoint.orientation.z,
                                    waypoint.orientation.w] for waypoint in waypoints])
            
            split_index = int(waypoints_np.shape[0] * 0.7)


            transformed_fine_waypoints = apply_transformation_to_waypoints(waypoints_np[split_index:], T_delta_world)
            fine_waypoints = [yumi.create_pose(*waypoint) for waypoint in transformed_fine_waypoints]


            yumi.group_l.set_pose_target(fine_waypoints[0])
            plan = yumi.group_l.plan()
            yumi.group_l.go(wait=True)
            rospy.sleep(1.5)
            (plan, fraction) = yumi.group_l.compute_cartesian_path(fine_waypoints, 0.01, 0.0)
            # AddTimeParameterization to better replicate demo dynamics
            plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.5, 0.5)

            display_trajectory(plan)

            yumi.group_l.execute(plan, wait=True)
            rospy.sleep(1)

            # Additional movement planning
            yumi.gripper_effort(yumi.LEFT, 20)
            move_upwards()

            yumi.reset_arm(yumi.LEFT)
            # User input to continue or break the loop
            user_input = input("Continue? (yes/no): ").lower()
            if user_input != 'yes':
                break

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Error in run function: {0}".format(e))


if __name__ == '__main__':
    rospy.init_node('yumi_main_loop')
    yumi.init_Moveit()
    gfk_left = GetFK('gripper_l_base', 'world')
    # args = parse_args()


    run(gfk_left)
