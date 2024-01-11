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
import warnings
import matplotlib.pyplot as plt

def smooth_trajectory(data, threshold):
    """
    Smooth the trajectory of a robot arm by keeping data points where at least one joint changes 
    by a value equal to or greater than the specified threshold from its previous state.

    :param data: List of lists containing the joint states of the robot arm.
    :param threshold: The minimum change in any joint state required to keep the data point.
    :return: List of lists containing the smoothed trajectory data.
    """
    smoothed_data = [data[0]]  # Initialize with the first data point

    for i in range(1, len(data)):
        # Check if any joint has changed at least the threshold
        if any(abs(data[i][j] - smoothed_data[-1][j]) >= threshold for j in range(len(data[i]))):
            smoothed_data.append(data[i])

    return smoothed_data


def run():
    rospy.init_node('yumi_traj_replay')

    # # Start by connecting to ROS and MoveIt!
    yumi.init_Moveit()

    with open("yumi_joint_states.json") as f:
        db = json.load(f)
    arm_data = []

    for data in db:
        # This check ensures that 'name' and 'position' exist and have the same length
        if "name" in data and "position" in data and len(data["name"]) == len(data["position"]):
            joint_data = dict(zip(data["name"], data["position"]))
            arm_data.append(joint_data)
    left_hand_data_2d = []

    # Desired order of joints
    joint_order = [1, 2, 7, 3, 4, 5, 6]

    for data in arm_data:
        left_hand_data = [data[f'yumi_joint_{i}_l'] for i in joint_order if f'yumi_joint_{i}_l' in data]
        left_hand_data_2d.append(left_hand_data)

    print("2D List of Left Hand Data in Specified Order:", left_hand_data_2d)

    # Apply the smoothing function with a chosen threshold (this threshold may need to be adjusted)
    threshold_value = 0.1
    smoothed_data = smooth_trajectory(left_hand_data_2d, threshold_value)
    print(smoothed_data)
    print(len(smoothed_data))
    # Plotting the smoothed trajectory
    # smoothed_joint_trajectories = list(zip(*smoothed_data))
    # plt.figure(figsize=(12, 8))
    # for i, joint_trajectory in enumerate(smoothed_joint_trajectories, start=1):
    #     plt.plot(joint_trajectory, label=f'Joint {i}')

    # plt.title('Smoothed Trajectory of Robot Arm Joints Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Joint States')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Get the robot's kinematic model
    # kinematic_model = robot.get_kinematic_model()
    # kinematic_state = robot.get_current_state()

    # # Set the joint states (assuming 'joint_states' is your data)
    # kinematic_state.set_joint_state_values(joint_states)

    # # Forward Kinematics
    # end_effector_pose = kinematic_state.get_end_effector_pose()
    # print("End Effector Pose:", end_effector_pose)

    # print(yumi.get_current_joint_values(yumi.LEFT))
    # yumi.go_to_joints(smoothed_data[-1], yumi.LEFT)
    for positions in smoothed_data:
        yumi.go_to_joints(positions, yumi.LEFT)

    rospy.spin()



if __name__ == '__main__':
    try:
        run()
        print("program_finished")
    except rospy.ROSInterruptException:
        pass