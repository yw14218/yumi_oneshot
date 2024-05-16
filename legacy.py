#!/usr/bin/env python3


import sys
import copy
import rospy
import moveit_commander
import yumi_moveit_utils as yumi
import moveit_msgs.msg
import geometry_msgs.msg
from std_srvs.srv import Empty
from trajectory_utils import pose_inv, translation_from_matrix, quaternion_from_matrix, create_homogeneous_matrix
import numpy as np


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




def run():
    """Starts the node

    Runs to start the node and initialize everthing. Runs forever via Spin()

    :returns: Nothing
    :rtype: None
    """

    rospy.init_node('yumi_moveit_demo')

    #Start by connecting to ROS and MoveIt!
    yumi.init_Moveit()

    # Reset YuMi joints to "home" position
    # yumi.reset_pose()

    # # yumi.reset_calib()
    yumi.reset_init()

    cur_pos_left = yumi.get_current_pose(yumi.LEFT)
    right = yumi.get_current_pose(yumi.RIGHT)
    print(cur_pos_left)
    print(right)

    # print(right)

    import threading
    # yumi.reset_init()

    # yumi.group_l.set_pose_target([
    #     0.40478858783062377,
    #     0.0822375992134779,
    #     0.5579262664335177,
    #     -0.9996890672397254,
    #     0.001056533957927872,
    #     0.020941860640910196,
    #     0.01349411168845638
    # ])

    # plan = yumi.group_l.plan()

    # yumi.group_r.set_pose_target([
    #     0.5329431807214543,
    #     -0.23440120550837626,
    #     0.46487699496824025,
    #     -0.9805945676315453,
    #     -0.04030389996510242,
    #     -0.01228235877075307,
    #     0.19146548838402241
    # ])

    # plan = yumi.group_r.plan()

    # yumi.group_r.go(wait=True)


    # def execute_trajectory(arm):
    #     # Assuming yumi.gripper_effort is a valid function call
    #     yumi.gripper_effort(arm, 15.0)

    # # Create threads, passing the function and its arguments correctly
    # thread_left = threading.Thread(target=execute_trajectory, args=(yumi.LEFT,))
    # thread_right = threading.Thread(target=execute_trajectory, args=(yumi.RIGHT,))

    # # Start threads
    # thread_left.start()
    # thread_right.start()

    # # Join threads to ensure both commands complete
    # thread_left.join()
    # thread_right.join()

    # # Drive YuMi end effectors to a desired position (pose_ee), and perform a grasping task with a given effort (grip_effort)
    # # Gripper effort: opening if negative, closing if positive, static if zero

    # pose_ee = [0.25, 0.34, 0.47, -3.14, 0, 0]
    # grip_effort = -10.0
    # move_and_grasp(yumi.LEFT, pose_ee, grip_effort)

    # pose_ee = [0.381, -0.0335, 0.378, -2.31087, -0.12899, 0.15397]
    # grip_effort = -10.0
    # move_and_grasp(yumi.RIGHT, pose_ee, grip_effort)


    # Print current joint angles
    # yumi.print_current_joint_states(yumi.RIGHT)
    # yumi.print_current_joint_states(yumi.LEFT)

    # back = [0.4118033296748138, 0.22378595991295266, 0.5793050897615559, 0.9934741742305934, -0.09152594162009686, 0.02502832203305602,0.06329020653785972]

    p = [5.81602659e-01, -4.11032233e-04,  4.34927401e-01, -0.05171688,  0.99592506, -0.01054581,  0.07312605]



    # p_world = [0.63028744, 0.01503754, 0.43909581, 0.00170821,  0.9979961 , -0.06246973,  0.00991924]

    # p_eef_new = [0.62040838, 0.08912966, 0.46735582, 0.99909779, -0.01761954,  0.01820735,  0.03408291]
    yumi.static_tf_broadcast("world", "goal", p)
    # yumi.static_tf_broadcast("world", "goal_inverse", _p)
    # # yumi.static_tf_broadcast("world", "goal_world", p_world)
    # yumi.group_l.set_pose_target(p_eef_new)

    # plan = yumi.group_l.plan()
    # yumi.group_l.go(wait=True)

    # T_camera_eef = np.load("handeye/T_C_EEF_wrist_l.npy")

    # delta_camera = create_homogeneous_matrix([-0.026914832310566607, -0.0010070975963056296, 0.015953726103354653], [-0.04317059, -0.0422042 , -0.15648559,  0.98583334])
    # T_new_eef_world = yumi.get_curent_T_left() @ T_camera_eef @ delta_camera @ pose_inv(T_camera_eef)
    # xyz = translation_from_matrix(T_new_eef_world).tolist()
    # quaternion = quaternion_from_matrix(T_new_eef_world).tolist()

    # yumi.static_tf_broadcast("world", "goal_world", xyz + quaternion)

    # # T_new_eef_world = T_camera_eef @ delta_camera @ pose_inv(T_camera_eef) @ yumi.get_curent_T_left()
    # # xyz = translation_from_matrix(T_new_eef_world).tolist()
    # # quaternion = quaternion_from_matrix(T_new_eef_world).tolist()

    # # yumi.static_tf_broadcast("world", "goal_world_reverse", xyz + quaternion)

    # yumi.group_l.set_pose_target(xyz+quaternion)
    # plan = yumi.group_l.plan()
    # yumi.group_l.go(wait=True)
    rospy.spin()




if __name__ == '__main__':
    try:
        run()
        print("program_finished")
    except rospy.ROSInterruptException:
        pass