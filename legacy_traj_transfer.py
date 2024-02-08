#!/usr/bin/env python3


import sys
import copy
import rospy
import moveit_commander
import yumi_moveit_utils as yumi
import moveit_msgs.msg
import geometry_msgs.msg
from std_srvs.srv import Empty
import json
import tf
import matplotlib.pyplot as plt
import copy
from get_fk import GetFK
from sensor_msgs.msg import JointState
from trajectory_utils import filter_joint_states, translation_from_matrix, quaternion_from_matrix
import numpy as np
import tf.transformations as tf_trans
from scipy.spatial.transform import Rotation as R

def cartesian(): # works!

    print("cur pose:")
    cur_pose = yumi.get_current_pose(yumi.LEFT)
    print(cur_pose)

    # We can get the name of the reference frame for this robot:
    planning_frame = yumi.group_l.get_planning_frame()
    print("============ Left Reference frame: {0}".format(planning_frame))
    planning_frame = yumi.group_r.get_planning_frame()
    print("============ Right Reference frame: {0}".format(planning_frame))
    current_frame = yumi.group_l.get_pose_reference_frame()
    print("Current pose reference frame for group_l:", current_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = yumi.group_l.get_end_effector_link()
    print("============ Left End effector: {0}".format(eef_link))
    eef_link = yumi.group_r.get_end_effector_link()
    print("============ Right End effector: {0}".format(eef_link))


def remove_duplicate_time_points(plan):
    new_points = []
    last_time_from_start = None
    for point in plan.joint_trajectory.points:
        if last_time_from_start is None or point.time_from_start != last_time_from_start:
            new_points.append(point)
            last_time_from_start = point.time_from_start
    plan.joint_trajectory.points = new_points
    return plan

def run():

    with open("split_lego_both.json") as f:
        joint_states = json.load(f)
    filtered_joint_states = filter_joint_states(joint_states, 0.1)

    msgs = [dict_to_joint_state(filtered_joint_state) for filtered_joint_state in filtered_joint_states]
    rospy.loginfo(type(msgs[0]))

    gfk_left = GetFK('gripper_l_base', 'world')
    eef_poses_left = [gfk_left.get_fk(msg) for msg in msgs]


    gfk_right = GetFK('gripper_r_base', 'world')
    eef_poses_right = [gfk_right.get_fk(msg) for msg in msgs]

    assert len(eef_poses_left) == len(msgs), "error in computing FK"
    assert len(eef_poses_right) == len(msgs), "error in computing FK"
    
    waypoints_left = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses_left]
    waypoints_right = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses_right]

    split_index = int(len(waypoints_left) * 0.6)

    yumi.go_to_joints(waypoints_left[0], yumi.LEFT)
    yumi.go_to_joints(waypoints_right[split_index], yumi.RIGHT)

    # for waypoint_right in waypoints_right:
    #     yumi.group_r.set_pose_target(waypoint_right)
    #     plan = yumi.group_r.plan()
    #     yumi.group_r.go(wait=True)


    # (plan, fraction) = yumi.group_l.compute_cartesian_path(
    #                             waypoints_left[1:],   # waypoints to follow
    #                             0.01,        # eef_step
    #                             0.0)         # jump_threshold

    # if (fraction == 1.0):
    #     plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.2, 0.2)
    #     plan = remove_duplicate_time_points(plan)

    # rospy.loginfo("Displaying trajectories")
    # # Initialize the display_trajectory_publisher
    # display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
    #                                            moveit_msgs.msg.DisplayTrajectory,
    #                                            queue_size=20)
    # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    # display_trajectory.trajectory_start = yumi.robot.get_current_state()
    # display_trajectory.trajectory.append(plan)
    # # Publish
    # display_trajectory_publisher.publish(display_trajectory)

    # yumi.group_l.execute(plan, wait=True)

    # rospy.sleep(1)
    # yumi.gripper_effort(yumi.LEFT, 20)
    # yumi.gripper_effort(yumi.RIGHT, -20)

    # # Compute the cartesian path
    # (plan, fraction) = yumi.group_r.compute_cartesian_path(
    #     waypoints_right[split_index:],  # waypoints to follow
    #     0.01,              # eef_step
    #     0.0)               # jump_threshold

    # # If the path computation was successful, retime and modify the trajectory
    # if fraction == 1.0:
    #     plan = yumi.group_r.retime_trajectory(yumi.robot.get_current_state(), plan, 0.2, 0.2)
    #     plan = remove_duplicate_time_points(plan)


    # rospy.loginfo("Displaying trajectories")
    # # Initialize the display_trajectory_publisher
    # display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
    #                                            moveit_msgs.msg.DisplayTrajectory,
    #                                            queue_size=20)
    # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    # display_trajectory.trajectory_start = yumi.robot.get_current_state()
    # display_trajectory.trajectory.append(plan)
    # # Publish
    # display_trajectory_publisher.publish(display_trajectory)

    # yumi.group_r.execute(plan, wait=True)

def dict_to_joint_state(data):
    """
    Convert a dictionary to a JointState message.

    Parameters:
    data (dict): A dictionary containing the joint state information.

    Returns:
    JointState: A ROS JointState message.
    """
    joint_state_msg = JointState()

    # Fill in the header information
    joint_state_msg.header.seq = data["header"]["seq"]
    joint_state_msg.header.stamp = rospy.Time(data["header"]["stamp"]["secs"], data["header"]["stamp"]["nsecs"])
    joint_state_msg.header.frame_id = data["header"]["frame_id"]

    # Fill in the joint names, positions, velocities, and efforts
    joint_state_msg.name = data["name"]
    joint_state_msg.position = data["position"]
    joint_state_msg.velocity = data["velocity"]
    joint_state_msg.effort = data["effort"]

    return joint_state_msg


if __name__ == '__main__':
    rospy.init_node('yumi_traj_replay')

    # # Start by connecting to ROS and MoveIt!
    yumi.init_Moveit()

    try:
        # cartesian()
        run()
        # computeFK()

        rospy.spin()
        print("program_finished")
    except rospy.ROSInterruptException:
        pass