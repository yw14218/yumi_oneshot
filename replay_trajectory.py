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
from movement_primitives.dmp import CartesianDMP
from scipy.spatial.transform import Rotation as R

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

def joint_trajectory_to_joint_states(joint_trajectory):
    """
    Convert a JointTrajectory message to a list of JointState messages.

    :param joint_trajectory: A JointTrajectory message.
    :return: A list of JointState messages corresponding to each point in the JointTrajectory.
    """
    joint_states = []

    for point in joint_trajectory.points:
        joint_state = JointState()
        joint_state.header = joint_trajectory.header  # Copy the header
        joint_state.name = joint_trajectory.joint_names  # Set the joint names
        joint_state.position = point.positions
        joint_state.velocity = point.velocities if point.velocities else []
        joint_state.effort = point.effort if point.effort else []
        joint_state.header.stamp = rospy.Time.now()  # Set the current time as the timestamp
        joint_states.append(joint_state)

    return joint_states

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

def move_upwards(distance = 0.15):
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
    wpose.position.z = initial_pose.position.z + distance
    wpose.orientation = copy.deepcopy(initial_pose.orientation)
    waypoints.append(copy.deepcopy(wpose))

    del waypoints[0]
    (plan, fraction) = yumi.group_l.compute_cartesian_path(waypoints, 0.01, 0.0)
    if fraction == 1.0:
        plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.05, 0.05)
    yumi.group_l.execute(plan, wait=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Process the file containing Yumi robot's joint states.")
    parser.add_argument('file_name', type=str, help='The name of the JSON file with joint states')
    return parser.parse_args()

def dmp(Y, T_delta_world = None):
    """
    Generate a trajectory with Cartesian DMP.
    Experiments show it may be a bad idea :(
    """
    # Prepare the time steps (T)
    n_steps = Y.shape[0]
    T = np.linspace(0, n_steps * 0.1, n_steps)  # Assuming each step is 0.1 seconds apart

    # Create a Cartesian DMP
    dt = 0.01
    execution_time = (n_steps - 1) * dt
    dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=10)

    # Train the DMP
    dmp.imitate(T, Y)

    new_start = Y[0].copy()

    if T_delta_world is None:
        new_goal = Y[-1].copy()
    else:
        new_goal = ptr.pqs_from_transforms(T_delta_world @ ptr.matrices_from_pos_quat(Y[-1].copy()))

    dmp.configure(start_y=new_start, goal_y=new_goal)

    # Generate a new trajectory towards the new goal
    _, Y = dmp.open_loop()

    return Y

def planKDL(Y, gfk_left, T_delta_world = None):
    """
    Generate a trajectory with MoveIt
    """
    from movement_primitives.kinematics import Kinematics

    with open("yumi_description/yumi.urdf", "r") as f:
        kin = Kinematics(f.read(), mesh_path="")

    # Creating the kinematic chain for the left arm
    left_arm_chain = kin.create_chain(
        ["yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", 
        "yumi_joint_3_l", "yumi_joint_4_l", "yumi_joint_5_l", 
        "yumi_joint_6_l"],
        "world", "gripper_l_base")  # Assuming 'yumi_tool0_l' is the end effector name for the left arm

    goal = ptr.pqs_from_transforms(T_delta_world @ ptr.matrices_from_pos_quat(Y[-1].copy()))
    yumi.group_l.set_pose_target(yumi.create_pose(*goal))
    coarse_plan = yumi.group_l.plan()
    joint_trajectory = coarse_plan[1].joint_trajectory
    positions = [point.positions for point in joint_trajectory.points]
    downsampled_positions = positions[::5]
    left_arm_transformations = [left_arm_chain.forward(qpos) for qpos in downsampled_positions]
    Y = [ptr.pqs_from_transforms(left_arm_transformation) for left_arm_transformation in left_arm_transformations]
    
    yumi.group_l.clear_pose_targets()
    return Y


def run(gfk_left):
    file_name="data/lego/lift_lego_left.json"

    with open(file_name) as f:
        joint_states = json.load(f)

    filtered_joint_states = filter_joint_states(joint_states, 0.01)
    msgs = [dict_to_joint_state(filtered_joint_state) for filtered_joint_state in filtered_joint_states]
    rospy.loginfo("{} waypoints in the trajectory".format(len(msgs)))

    eef_poses_left = [gfk_left.get_fk(msg) for msg in msgs]
    assert len(eef_poses_left) == len(msgs), "Error in computing FK"

    # Planning and executing transferred trajectories
    T_delta_world = np.array([
        [ 0.79857538, -0.60179267, -0.011088  ,  0.28067213],
        [ 0.60165335,  0.79864132, -0.01361215, -0.38310653],
        [ 0.01704703,  0.00419919,  0.99984587, -0.00643008],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])

    # Planning and executing trajectories
    waypoints = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses_left]
    waypoints_np = np.array([[waypoint.position.x, waypoint.position.y, waypoint.position.z,
                            waypoint.orientation.x, waypoint.orientation.y, waypoint.orientation.z,
                            waypoint.orientation.w] for waypoint in waypoints])
    
    split_index = int(waypoints_np.shape[0] * 0.7)

    # choose one
    Y = dmp(waypoints_np[:split_index], T_delta_world)
    # Y = planKDL(waypoints_np[:split_index], gfk_left, T_delta_world)

    transformed_fine_waypoints = apply_transformation_to_waypoints(waypoints_np[split_index:], T_delta_world)
    fine_waypoints = [yumi.create_pose(*waypoint) for waypoint in transformed_fine_waypoints]

    coarse_waypoints = [yumi.create_pose(*waypoint) for waypoint in Y][::2] # downsample

    whole_waypoints = coarse_waypoints + fine_waypoints
    print(type(fine_waypoints[0]))

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
    yumi.gripper_effort(yumi.LEFT, 15)
    move_upwards()



if __name__ == '__main__':
    rospy.init_node('yumi_replay_trajectory')
    yumi.init_Moveit()
    gfk_left = GetFK('gripper_l_base', 'world')
    # args = parse_args()

    try:
        run(gfk_left)
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupted")
