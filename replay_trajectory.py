#!/usr/bin/env python3

import rospy
import copy
import json
import argparse
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import yumi_moveit_utils as yumi
from get_fk import GetFK
from sensor_msgs.msg import JointState
from trajectory_utils import filter_joint_states, apply_transformation_to_waypoints

def display_info():
    try:
        # Displaying planning frames for both left and right arms
        planning_frame_left = yumi.group_l.get_planning_frame()
        rospy.loginfo("Left Reference frame: {0}".format(planning_frame_left))
        planning_frame_right = yumi.group_r.get_planning_frame()
        rospy.loginfo("Right Reference frame: {0}".format(planning_frame_right))

        # Displaying current pose reference frames
        current_frame_left = yumi.group_l.get_pose_reference_frame()
        rospy.loginfo("Current pose reference frame for left arm: {0}".format(current_frame_left))
        current_frame_right = yumi.group_r.get_pose_reference_frame()
        rospy.loginfo("Current pose reference frame for right arm: {0}".format(current_frame_right))

        # Displaying end-effector links
        eef_link_left = yumi.group_l.get_end_effector_link()
        rospy.loginfo("Left End effector: {0}".format(eef_link_left))
        eef_link_right = yumi.group_r.get_end_effector_link()
        rospy.loginfo("Right End effector: {0}".format(eef_link_right))
    except Exception as e:
        rospy.logerr("Error in display_info: {0}".format(e))

def dict_to_joint_state(data):
    """
    Convert a dictionary to a JointState message.
    """
    joint_state_msg = JointState()
    joint_state_msg.header.seq = data["header"]["seq"]
    joint_state_msg.header.stamp = rospy.Time(data["header"]["stamp"]["secs"], data["header"]["stamp"]["nsecs"])
    joint_state_msg.header.frame_id = data["header"]["frame_id"]
    joint_state_msg.name = data["name"]
    joint_state_msg.position = data["position"]
    joint_state_msg.velocity = data["velocity"]
    joint_state_msg.effort = data["effort"]
    return joint_state_msg

def display_trajectory(plan):
    """
    Display trajectory in RViz.
    """
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = yumi.robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    display_trajectory_publisher.publish(display_trajectory)

def move_upwards():
    """
    Plan and execute a movement upwards.
    """
    waypoints = []
    initial_pose = yumi.group_l.get_current_pose().pose
    waypoints.append(copy.deepcopy(initial_pose))

    # Adjusting the position for the upward movement
    wpose = geometry_msgs.msg.Pose()
    wpose.position.x = initial_pose.position.x
    wpose.position.y = initial_pose.position.y
    wpose.position.z = initial_pose.position.z + 0.2
    wpose.orientation = copy.deepcopy(initial_pose.orientation)
    waypoints.append(copy.deepcopy(wpose))

    (plan, fraction) = yumi.group_l.compute_cartesian_path(waypoints, 0.01, 0.0)
    if fraction == 1.0:
        plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.05, 0.05)
    yumi.group_l.execute(plan, wait=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Process the file containing Yumi robot's joint states.")
    parser.add_argument('file_name', type=str, help='The name of the JSON file with joint states')
    return parser.parse_args()


def run(file_name):
    try:
        with open(file_name) as f:
            joint_states = json.load(f)

        filtered_joint_states = filter_joint_states(joint_states, 0.1)
        msgs = [dict_to_joint_state(filtered_joint_state) for filtered_joint_state in filtered_joint_states]
        rospy.loginfo("{} waypoints in the trajectory".format(len(msgs)))

        gfk_left = GetFK('gripper_l_base', 'world')
        eef_poses_left = [gfk_left.get_fk(msg) for msg in msgs]
        assert len(eef_poses_left) == len(msgs), "Error in computing FK"

        # Planning and executing trajectories
        waypoints = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses_left]

        # Planning and executing transferred trajectories
        # delta_R = np.array([
        #     [0.98628563,  0.16504762,  0.,          0.07052299],
        #     [-0.16504762, 0.98628563,  0.,          0.0126316 ],
        #     [0.,          0.,          1.,         -0.00986874],
        #     [0.,          0.,          0.,          1.        ]
        # ])
        # transformed_waypoints = apply_transformation_to_waypoints(eef_poses_left, delta_R)
        # new_waypoints = [yumi.create_pose(*waypoint) for waypoint in transformed_waypoints]
        # rospy.loginfo(new_waypoints)

        (plan, fraction) = yumi.group_l.compute_cartesian_path(waypoints, 0.01, 0.0)
        # plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.1, 0.1)
        display_trajectory(plan)

        yumi.group_l.execute(plan, wait=True)
        rospy.sleep(1)

        # # Additional movement planning
        # yumi.gripper_effort(yumi.LEFT, 15)
        # move_upwards()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Error in run function: {0}".format(e))


if __name__ == '__main__':
    rospy.init_node('yumi_replay_trajectory')
    yumi.init_Moveit()

    args = parse_args()

    try:
        run(args.file_name)
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupted")
