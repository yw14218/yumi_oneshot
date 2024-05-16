#!/usr/bin/env python3
import rospy
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_of_parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_of_parent_dir)

import yumi_moveit_utils as yumi
from base_experiment import YuMiExperiment

class RiceCookerExperiment(YuMiExperiment):
    
    def replay(self, live_waypoints):

        live_bottleneck_left, live_bottleneck_right, \
        live_push_left, live_release_right, \
        live_close_left_start, live_close_left_end = live_waypoints
        
        """
        Move to the bottlenecks
        """
        yumi.plan_both_arms(live_bottleneck_left, live_bottleneck_right)

        yumi.close_grippers(yumi.LEFT)

        """
        Push button and retreat
        """
        (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_push_left)], 0.01, 0.0)
        yumi.group_l.execute(plan_left)
        rospy.sleep(0.1)

        yumi.plan_left_arm(live_bottleneck_left)

        """
        Put rice in pot
        """
        (plan_right, _) = yumi.group_r.compute_cartesian_path([yumi.create_pose(*live_release_right)], 0.01, 0.0)
        yumi.group_r.execute(plan_right)
        rospy.sleep(0.1)

        yumi.open_grippers(yumi.RIGHT)
        rospy.sleep(0.1)
        yumi.close_grippers(yumi.RIGHT)

        """
        Move to the close lid pose
        """
        (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_close_left_start)], 0.01, 0.0)
        (plan_right, _) = yumi.group_r.compute_cartesian_path([yumi.create_pose(*live_bottleneck_right)], 0.01, 0.0)

        yumi.group_r.execute(plan_right)
        rospy.sleep(0.1)
        
        yumi.group_l.execute(plan_left)
        rospy.sleep(0.1)

        """
        Close lid
        """
        (plan_left, _) = yumi.group_l.compute_cartesian_path([yumi.create_pose(*live_close_left_end)], 0.01, 0.0)
        yumi.group_r.execute(plan_left)

if __name__ == '__main__':
    try:
        rospy.init_node('yumi_moveit_demo')
        yumi.init_Moveit()

        MODE = "REPLAY"
        scissorExperiment = RiceCookerExperiment("experiments/rice_cooker", "rice cooker", MODE)
        yumi.reset_init()
        scissorExperiment.run()
        rospy.spin()
        
    except Exception as e:
        print(f"Error: {e}")