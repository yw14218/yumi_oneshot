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
import copy

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

def cartesian():

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

    # pose = copy.deepcopy(cur_pose.pose)
    # x,y,z,q1,q2,q3,q4 =  pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    # waypoints = [x+0.1,y,z,q1,q2,q3,q4]

    # this code makes the robot move??? TODO: fix
    waypoints = []

    # start with the current pose
    print(yumi.group_l.get_current_pose())
    waypoints.append(yumi.group_l.get_current_pose().pose)

    # first orient gripper and move forward (+x)
    wpose = geometry_msgs.msg.Pose()
    wpose.position.x = waypoints[0].position.x  # x is up in the gripper_l_base frame
    wpose.position.y = waypoints[0].position.y  
    wpose.position.z = waypoints[0].position.z 
    wpose.orientation.x = waypoints[0].orientation.x
    wpose.orientation.y = waypoints[0].orientation.y
    wpose.orientation.z = waypoints[0].orientation.z
    wpose.orientation.w = waypoints[0].orientation.w
    waypoints.append(copy.deepcopy(wpose))

    # print(waypoints)
    (plan, fraction) = yumi.group_l.compute_cartesian_path(
                                waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                0.0)         # jump_threshold
    print(plan)
    # Initialize the display_trajectory_publisher
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = yumi.robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    # Publish
    display_trajectory_publisher.publish(display_trajectory)

    # yumi.group_l.execute(plan, wait=True)
    rospy.sleep(3)
    print("============ Waiting ...")

def run():

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

    # with open("yumi_left_2d.json", "w") as file:
    #     json.dump(left_hand_data_2d, file)

    # Apply the smoothing function with a chosen threshold (this threshold may need to be adjusted)
    threshold_value = 0.1
    smoothed_data = smooth_trajectory(left_hand_data_2d, threshold_value)
    print(smoothed_data)
    print(len(smoothed_data))


    # # Plotting the smoothed trajectory
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

    # yumi.go_to_joints(smoothed_data[-1], yumi.LEFT)

    # for positions in smoothed_data:
    #     yumi.go_to_joints(positions, yumi.LEFT)



if __name__ == '__main__':
    rospy.init_node('yumi_traj_replay')

    # # Start by connecting to ROS and MoveIt!
    yumi.init_Moveit()

    try:
        cartesian()
        # run()

        rospy.spin()
        print("program_finished")
    except rospy.ROSInterruptException:
        pass