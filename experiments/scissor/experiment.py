#!/usr/bin/env python3
import rospy
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_of_parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_of_parent_dir)

import yumi_moveit_utils as yumi
from base_experiment import YuMiExperiment

class ScissorExperiment(YuMiExperiment):
    
    def replay(self, live_waypoints):

        live_bottleneck_left, live_bottleneck_right, \
        live_grasp_left, live_grasp_right, \
        live_lift_left = live_waypoints
        
        """
        Move to the bottlenecks
        """
        yumi.plan_both_arms(live_bottleneck_left, live_bottleneck_right)

        rospy.sleep(0.1)
        yumi.close_grippers(yumi.RIGHT)

        """
        Cartesian trajectories to reach the grasp pose
        """
        live_grasp_right[2] += 0.01
        # (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_grasp_left)], 0.01, 0.0)
        (plan_right, _) = yumi.group_r.compute_cartesian_path([yumi.create_pose(*live_grasp_right)], 0.01, 0.0)

        # # # Align the trajectories
        # # align_trajectories(plan_left, plan_right)
        # # # Merge them
        # # merged_plan = merge_trajectories(plan_left, plan_right)
        # # yumi.group_both.execute(merged_plan)

        yumi.group_r.execute(plan_right)
        rospy.sleep(0.1)
        # yumi.gripper_effort(yumi.RIGHT, -20.0)

        # yumi.group_l.execute(plan_left)
        # rospy.sleep(0.1)
        # yumi.gripper_effort(yumi.LEFT, 20.0)

        # # """
        # # Operate the grippers simultaneously
        # # """
        # # rospy.sleep(0.2)

        # # yumi.close_left_open_right_in_threads([yumi.LEFT, yumi.RIGHT])

        # rospy.sleep(0.2)

        # """
        # Uncovering trajectories
        # """
        
        # (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_lift_left)], 0.01, 0.0)
        # yumi.group_l.execute(plan_left)

if __name__ == '__main__':
    try:
        rospy.init_node('yumi_moveit_demo')
        yumi.init_Moveit()

        MODE = "HEADCAM"
        # MODE = "DINOBOT"
        scissorExperiment = ScissorExperiment("experiments/scissor", "black scissor", MODE)
        yumi.reset_init()
        scissorExperiment.run()
        rospy.spin()
        
    except Exception as e:
        print(f"Error: {e}")