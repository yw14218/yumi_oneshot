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

def run():

    with open("yumi_joint_states.json") as f:
        joint_states = json.load(f)
    filtered_joint_states = filter_joint_states(joint_states, 0.1)

    msgs = [dict_to_joint_state(filtered_joint_state) for filtered_joint_state in filtered_joint_states]
    rospy.loginfo(type(msgs[0]))
    print(len(msgs))
    gfk = GetFK('gripper_l_base', 'world')
    eef_poses = [gfk.get_fk(msg) for msg in msgs]
    assert len(eef_poses) == len(msgs), "error in computing FK"
    print(len(eef_poses))

    # yumi.go_to_joints(smoothed_data[-1], yumi.LEFT)

    # for positions in smoothed_data:
    #     yumi.go_to_joints(positions, yumi.LEFT)
    waypoints = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses]
    waypoint_nums = [[waypoint.position.x, waypoint.position.y, waypoint.position.z, 
                        waypoint.orientation.x, waypoint.orientation.y, waypoint.orientation.z, 
                        waypoint.orientation.w] for waypoint in waypoints]
    
    delta_R = np.array([
        [0.98628563,  0.16504762,  0.,          0.07052299],
        [-0.16504762, 0.98628563,  0.,          0.0126316 ],
        [0.,          0.,          1.,         -0.00986874],
        [0.,          0.,          0.,          1.        ]
    ])


    transformed_waypoints = []
    # Apply the transformation to each waypoint
    for waypoint in waypoint_nums:
        waypoint_translation = waypoint[:3]
        waypoint_rotation = R.from_quat(waypoint[3:7]).as_matrix()

        RW = np.eye(4)
        RW[:3, :3] = waypoint_rotation
        RW[:3, 3] = waypoint_translation

        # Apply the composite transformation
        transformed_matrix = delta_R @ RW
        transformed_translation = translation_from_matrix(transformed_matrix)
        transformed_quaternion = quaternion_from_matrix(transformed_matrix)

        transformed_waypoints.append(np.concatenate((transformed_translation, transformed_quaternion)).tolist())

    new_waypoints = [yumi.create_pose(*waypoint) for waypoint in transformed_waypoints]
    print(new_waypoints)

    (plan, fraction) = yumi.group_l.compute_cartesian_path(
                                new_waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                0.0)         # jump_threshold

    # if (fraction == 1.0):
    #     plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.05, 0.05)

    rospy.loginfo("Displaying trajectories")
    # Initialize the display_trajectory_publisher
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = yumi.robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    # Publish
    display_trajectory_publisher.publish(display_trajectory)

    print(plan)
    yumi.group_l.execute(plan, wait=True)
    rospy.sleep(1)
    yumi.gripper_effort(yumi.LEFT, 15)

    waypoints = []

    waypoints.append(yumi.group_l.get_current_pose().pose)

    # first orient gripper and move forward (+x)
    wpose = geometry_msgs.msg.Pose()
    wpose.position.x = waypoints[0].position.x
    wpose.position.y = waypoints[0].position.y  
    wpose.position.z = waypoints[0].position.z + 0.2
    wpose.orientation.x = waypoints[0].orientation.x
    wpose.orientation.y = waypoints[0].orientation.y
    wpose.orientation.z = waypoints[0].orientation.z
    wpose.orientation.w = waypoints[0].orientation.w
    waypoints.append(copy.deepcopy(wpose))

    del waypoints[0]
    # print(waypoints)
    (plan, fraction) = yumi.group_l.compute_cartesian_path(
                                waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                0.0)         # jump_threshold
    # print(plan)
    if (fraction == 1.0):
        plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.05, 0.05)
    yumi.group_l.execute(plan, wait=True)

    rospy.sleep(1)

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