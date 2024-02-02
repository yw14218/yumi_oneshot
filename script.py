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
from replay_trajectory import dict_to_joint_state
import threading
from ikSolver import IKSolver

def close_grippers(arm):
    """Closes the grippers.

    Closes the grippers with an effort of 15 and then relaxes the effort to 0.

    :param arm: The side to be closed (moveit_utils LEFT or RIGHT)
    :type arm: int
    :returns: Nothing
    :rtype: None
    """
    yumi.gripper_effort(arm, 15.0)
    yumi.gripper_effort(arm, 0.0)

def open_grippers(arm):
    """Opens the grippers.

    Opens the grippers with an effort of -15 and then relaxes the effort to 0.

    :param arm: The side to be opened (moveit_utils LEFT or RIGHT)
    :type arm: int
    :returns: Nothing
    :rtype: None
    """
    yumi.gripper_effort(arm, -15.0)
    yumi.gripper_effort(arm, 0.0)

def move_and_grasp(arm, pose_ee, grip_effort):
    try:
        yumi.traverse_path([pose_ee], arm, 10)
    except Exception:
        if (arm == yumi.LEFT):
            yumi.plan_and_move(yumi.group_l, yumi.create_pose_euler(pose_ee[0], pose_ee[1], pose_ee[2], pose_ee[3], pose_ee[4], pose_ee[5]))
        elif (arm == yumi.RIGHT):
            yumi.plan_and_move(yumi.group_r, yumi.create_pose_euler(pose_ee[0], pose_ee[1], pose_ee[2], pose_ee[3], pose_ee[4], pose_ee[5]))

    if (grip_effort <= 20 and grip_effort >= -20):
        yumi.gripper_effort(arm, grip_effort)
    else:
        print("The gripper effort values should be in the range [-20, 20]")

def get_waypoints(path, eef):
    with open(path) as f:
        joint_states = json.load(f)
    filtered_joint_states = filter_joint_states(joint_states, 0.1)

    msgs = [dict_to_joint_state(filtered_joint_state) for filtered_joint_state in filtered_joint_states]
    rospy.loginfo(type(msgs[0]))
    print(len(msgs))
    gfk = GetFK(eef, 'world')
    eef_poses = [gfk.get_fk(msg) for msg in msgs]
    assert len(eef_poses) == len(msgs), "error in computing FK"

    waypoints = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses]

    return waypoints

def run():

    rospy.init_node('yumi_moveit_demo')
    yumi.init_Moveit()

    yumi.reset_init()


    gfk_left = GetFK('gripper_l_base', 'world')
    gfk_right = GetFK('gripper_r_base', 'world')

    file_name="split_lego_both.json"

    with open(file_name) as f:
        joint_states = json.load(f)

    filtered_joint_states = filter_joint_states(joint_states, 0.01)
    msgs = [dict_to_joint_state(filtered_joint_state) for filtered_joint_state in filtered_joint_states]
    rospy.loginfo("{} waypoints in the trajectory".format(len(msgs)))

    eef_poses_left = [gfk_left.get_fk(msg) for msg in msgs]
    eef_poses_right = [gfk_right.get_fk(msg) for msg in msgs]
    assert len(eef_poses_left) == len(msgs), "Error in computing FK"

    waypoints_left = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses_left]
    waypoints_right = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses_right]

    del waypoints_right[0]
    del waypoints_left[0]

    left_eef = "gripper_l_base"
    right_eef = "gripper_r_base"

    ik_solver_left = IKSolver(group_name="left_arm", ik_link_name=left_eef)
    ik_solver_right = IKSolver(group_name="right_arm", ik_link_name=right_eef)
    
    
    # Example coordinates and orientation (Euler angles in radians)
    ik_left = ik_solver_left.get_ik(waypoints_left[0]).solution.joint_state.position
    ik_right = ik_solver_right.get_ik(waypoints_right[0]).solution.joint_state.position

    yumi.group_both.set_joint_value_target(ik_left[:7] + ik_right[9:16])
    yumi.group_both.go(wait=True)

    (plan, fraction) = yumi.group_l.compute_cartesian_path(waypoints_left, 0.01, 0.0)

    # AddTimeParameterization to better replicate demo dynamics
    plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.5, 0.5)


    yumi.group_l.execute(plan, wait=True)

    (plan, fraction) = yumi.group_r.compute_cartesian_path(waypoints_right, 0.01, 0.0)

    # AddTimeParameterization to better replicate demo dynamics
    plan = yumi.group_r.retime_trajectory(yumi.robot.get_current_state(), plan, 0.5, 0.5)

    yumi.group_r.execute(plan, wait=True)
    


    # print(waypoints_left[-5])
    # print(waypoints_right[1])
    # # yumi.group_both.set_pose_target(waypoints_left[-5], end_effector_link=left_eef)
    # yumi.group_both.set_pose_target(waypoints_left[-5], end_effector_link=right_eef)
    # plan = yumi.group_both.plan()
    # print(plan)
    # yumi.group_both.go(wait=True)

    # yumi.group_r.set_pose_target(waypoints_right[0])
    # plan = yumi.group_r.plan()
    # yumi.group_r.go(wait=False)

    # yumi.group_l.set_pose_target(waypoints_left[-1])
    # plan = yumi.group_l.plan()
    # yumi.group_l.go(wait=False)

    # (plan, fraction) = yumi.group_l.compute_cartesian_path(
    #                             waypoints,   # waypoints to follow
    #                             0.01,        # eef_step
    #                             0.0)         # jump_threshold
    
    # plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.3, 0.3)

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

    # yumi.group_l.execute(plan, wait=False)

    # rospy.sleep(1)
    # yumi.gripper_effort(yumi.LEFT, 20)

    # waypoints = []

    # waypoints.append(yumi.group_l.get_current_pose().pose)

    # # first orient gripper and move forward (+x)
    # wpose = geometry_msgs.msg.Pose()
    # wpose.position.x = waypoints[0].position.x
    # wpose.position.y = waypoints[0].position.y  
    # wpose.position.z = waypoints[0].position.z + 0.12
    # wpose.orientation.x = waypoints[0].orientation.x
    # wpose.orientation.y = waypoints[0].orientation.y
    # wpose.orientation.z = waypoints[0].orientation.z
    # wpose.orientation.w = waypoints[0].orientation.w
    # waypoints.append(copy.deepcopy(wpose))

    # del waypoints[0]
    # # print(waypoints)
    # (plan, fraction) = yumi.group_l.compute_cartesian_path(
    #                             waypoints,   # waypoints to follow
    #                             0.01,        # eef_step
    #                             0.0)         # jump_threshold
    # if (fraction == 1.0):
    #     plan = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan, 0.3, 0.3)
    # yumi.group_l.execute(plan, wait=False)






    rospy.spin()




if __name__ == '__main__':
    
    run()
    print("program_finished")
