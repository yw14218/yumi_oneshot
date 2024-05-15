#!/usr/bin/env python3
import json
import sys
import rospy
import numpy as np
import yumi_moveit_utils as yumi
from poseEstimation import PoseEstimation
from trajectory_utils import apply_transformation_to_waypoints,align_trajectories, merge_trajectories, \
                             create_homogeneous_matrix, pose_inv, quaternion_from_matrix
from dinobotAlignment import DINOBotAlignment
from base_experiment import YuMiExperiment

class ScissorExperiment(YuMiExperiment):
    def replay(self, live_waypoints):

        live_bottleneck_left, live_bottleneck_right, \
        live_grasp_left, live_grasp_right, \
        live_lift_left = live_waypoints
        
        """
        Move to the bottlenecks
        """
        self.yumi.plan_both_arms(live_bottleneck_left, live_bottleneck_right)

        yumi.close_grippers(yumi.RIGHT)

        """
        Cartesian trajectories to reach the grasp pose
        """
        (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_grasp_left)], 0.01, 0.0)
        (plan_right, _) = yumi.group_r.compute_cartesian_path([yumi.create_pose(*live_grasp_right)], 0.01, 0.0)

        # # Align the trajectories
        # align_trajectories(plan_left, plan_right)
        # # Merge them
        # merged_plan = merge_trajectories(plan_left, plan_right)
        # yumi.group_both.execute(merged_plan)

        yumi.group_r.execute(plan_right)
        rospy.sleep(0.1)
        yumi.gripper_effort(yumi.RIGHT, -20.0)

        yumi.group_l.execute(plan_left)
        rospy.sleep(0.1)
        yumi.gripper_effort(yumi.LEFT, 20.0)

        """
        Operate the grippers simultaneously
        """
        rospy.sleep(0.25)

        yumi.close_left_open_right_in_threads([yumi.LEFT, yumi.RIGHT])

        rospy.sleep(0.25)

        """
        Uncovering trajectories
        """
        
        yumi.group_l.set_pose_target(live_lift_left)
        plan = yumi.group_l.plan()
        yumi.group_l.go(wait=True)
        yumi.group_l.stop()
        yumi.group_l.clear_pose_targets()

if __name__ == '__main__':
    try:
        rospy.init_node('yumi_moveit_demo')
        yumi.init_Moveit()

        scissorExperiment = ScissorExperiment("scissor", "black scissor", "REPLAY", yumi)
        scissorExperiment.run()
        rospy.spin()
        
    except Exception as e:
        print(f"Error: {e}")