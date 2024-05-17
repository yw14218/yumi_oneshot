#!/usr/bin/env python3
import rospy
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_of_parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_of_parent_dir)

import yumi_moveit_utils as yumi
from base_experiment import YuMiExperiment
from trajectory_utils import align_trajectory_points, merge_trajectories

class WoodExperiment(YuMiExperiment):
    
    def replay(self, live_waypoints):

        live_bottleneck_left, live_bottleneck_right, \
        live_grasp_left, live_grasp_right, \
        live_lift_left, live_lift_right = live_waypoints
        
        """
        Move to the bottlenecks
        """
        yumi.plan_both_arms(live_bottleneck_left, live_bottleneck_right)

        """
        Cartesian trajectories to reach the grasp pose
        """
        (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_grasp_left)], 0.01, 0.0)
        (plan_right, _) = yumi.group_r.compute_cartesian_path([yumi.create_pose(*live_grasp_right)], 0.01, 0.0)

        plan_left, plan_right = align_trajectory_points(plan_left, plan_right)
        merged_plan = merge_trajectories(plan_left, plan_right)
        merged_plan = yumi.group_both.retime_trajectory(yumi.robot.get_current_state(), merged_plan, 0.5, 0.5)
        yumi.group_both.execute(merged_plan)

        # yumi.group_r.execute(plan_right)
        # rospy.sleep(0.2)

        # # yumi.group_l.execute(plan_left)
        # # rospy.sleep(0.2)

        """
        Close the grippers simultaneously
        """
        yumi.close_gripper_in_threads([yumi.LEFT, yumi.RIGHT])

        rospy.sleep(0.1)

        """
        Lifting trajectories
        """
        yumi.plan_both_arms(live_lift_left, live_lift_right)

if __name__ == '__main__':
    try:
        rospy.init_node('yumi_wood_experiment')
        yumi.init_Moveit()

        MODE = "HEADCAM"
        scissorExperiment = WoodExperiment("experiments/wood", "wood stand", MODE)
        yumi.reset_init()
        scissorExperiment.run()
        rospy.spin()
        
    except Exception as e:
        print(f"Error: {e}")