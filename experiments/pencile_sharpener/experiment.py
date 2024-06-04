#!/usr/bin/env python3
import rospy
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_of_parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_of_parent_dir)

import yumi_moveit_utils as yumi
from base_experiment import YuMiExperiment
from trajectory_utils import compute_pre_grasp_pose

class SharpenerExperiment(YuMiExperiment):
    
    @staticmethod
    def replay(live_waypoints):

        live_bottleneck_left, insert_left_start,\
        live_grasp_right, insert_left_end = live_waypoints
        

        """
        Move to the bottlenecks
        """
        yumi.gripper_effort(yumi.LEFT, 20)
        yumi.plan_left_arm(live_bottleneck_left)

        right_pre_grasp_pose = compute_pre_grasp_pose(live_grasp_right[:3], live_grasp_right[3:]).tolist()
        # yumi.plan_right_arm(right_pre_grasp_pose)

        """
        Cartesian trajectories to reach the grasp pose
        """
        
        # (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*insert_left_start)], 0.01, 0.0)
        # yumi.group_l.execute(plan_left)

        """
        Uncovering trajectories
        """
        (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*insert_left_start), yumi.create_pose(*insert_left_end)], 0.01, 0.0)
        plan_left = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan_left, 0.025, 0.025)
        yumi.group_l.execute(plan_left)

        # (plan_right, _) = yumi.group_r.compute_cartesian_path([yumi.create_pose(*live_grasp_right)], 0.01, 0.0)
        # plan_right = yumi.group_r.retime_trajectory(yumi.robot.get_current_state(), plan_right, 0.05, 0.05)
        # yumi.group_r.execute(plan_right)

        rospy.sleep(0.1)
        # yumi.close_grippers(yumi.RIGHT)

        yumi.open_grippers(yumi.LEFT)

    @staticmethod
    def reset():
        yumi.operate_gripper_in_threads([yumi.LEFT, yumi.RIGHT], close=False)
        yumi.reset_init()

if __name__ == '__main__':
    try:
        rospy.init_node('yumi_moveit_demo')
        yumi.init_Moveit()

        MODE = "REPLAY"
        # MODE = "DINOBOT"
        scissorExperiment = SharpenerExperiment("experiments/pencile_sharpener", "blue pencile sharpener", MODE)
        scissorExperiment.reset()

        user_input = input("Continue? (yes/no): ").lower()
        scissorExperiment.run()
        rospy.spin()
        
    except Exception as e:
        print(f"Error: {e}")