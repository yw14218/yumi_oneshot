#!/usr/bin/env python3
import rospy
import sys
import os
import moveit_utils.yumi_moveit_utils as yumi
from base_experiment import YuMiExperiment

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_of_parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_of_parent_dir)

class ScissorExperiment(YuMiExperiment):
    
    @staticmethod
    def replay(live_waypoints, arm=yumi.RIGHT):

        live_bottleneck_left, live_bottleneck_right, \
        live_grasp_left, live_grasp_right, \
        live_lift_left = live_waypoints
        
        """
        Move to the bottlenecks
        """

        if arm is None:
            yumi.plan_both_arms(live_bottleneck_left, live_bottleneck_right)
        elif arm == yumi.RIGHT:
            yumi.plan_right_arm(live_bottleneck_right)
        elif arm == yumi.LEFT:
            yumi.plan_left_arm(live_bottleneck_left)
        else:
            raise NotImplementedError

        """
        Cartesian trajectories to reach the grasp pose
        """
        
        (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_grasp_left)], 0.01, 0.0)
        (plan_right, _) = yumi.group_r.compute_cartesian_path([yumi.create_pose(*live_grasp_right)], 0.01, 0.0)

        yumi.close_grippers(yumi.RIGHT)

        yumi.group_r.execute(plan_right)
        rospy.sleep(0.1)
        yumi.gripper_effort(yumi.RIGHT, -20.0)

        yumi.group_l.execute(plan_left)
        rospy.sleep(0.1)
        yumi.gripper_effort(yumi.LEFT, 20.0)

        """
        Uncovering trajectories
        """
        (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_lift_left)], 0.01, 0.0)
        plan_left = yumi.group_l.retime_trajectory(yumi.robot.get_current_state(), plan_left, 0.5, 0.5)
        yumi.group_l.execute(plan_left)

    @staticmethod
    def reset():
        yumi.close_grippers(yumi.RIGHT)
        yumi.open_grippers(yumi.LEFT)
        yumi.reset_init()

if __name__ == '__main__':
    try:
        rospy.init_node('yumi_moveit_demo')
        yumi.init_Moveit()

        MODE = "REPLAY"
        # MODE = "DINOBOT"
        scissorExperiment = ScissorExperiment("experiments/scissor", "black scissor", MODE)
        yumi.reset_init()
        scissorExperiment.run()
        rospy.spin()
        
    except Exception as e:
        print(f"Error: {e}")