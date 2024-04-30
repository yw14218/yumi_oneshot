#!/usr/bin/env python3


import sys
import copy
import rospy
import moveit_commander
import yumi_moveit_utils as yumi
import moveit_msgs.msg
import geometry_msgs.msg
from std_srvs.srv import Empty




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
    # yumi.reset_init()

    left = yumi.get_current_pose(yumi.LEFT)
    right = yumi.get_current_pose(yumi.RIGHT)


    print(left)
    print(right)


    yumi.reset_init()

    yumi.group_l.set_pose_target([
        0.40478858783062377,
        0.0822375992134779,
        0.5579262664335177,
        -0.9996890672397254,
        0.001056533957927872,
        0.020941860640910196,
        0.01349411168845638
    ])

    plan = yumi.group_l.plan()
    yumi.group_l.go(wait=True)

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
    yumi.print_current_joint_states(yumi.LEFT)
    
    rospy.spin()




if __name__ == '__main__':
    try:
        run()
        print("program_finished")
    except rospy.ROSInterruptException:
        pass