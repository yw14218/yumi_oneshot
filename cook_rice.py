#!/usr/bin/env python3
import json
import sys
import rospy
import numpy as np
import yumi_moveit_utils as yumi
from poseEstimation import PoseEstimation
from trajectory_utils import apply_transformation_to_waypoints,align_trajectory_points, merge_trajectories, \
                             create_homogeneous_matrix, pose_inv, quaternion_from_matrix
from dinobotAlignment import DINOBotAlignment

DIR = "data/rice_cooker"
OBJECT = "rice cooker"

def replay(live_waypoints):

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


def run(dbn, MODE):

    keys = ["bottleneck_left", "bottleneck_right", 
            "push_left", "release_right", 
            "close_left_start", "close_left_end"]
    demo_waypoints = np.vstack([dbn[key] for key in keys])

    if MODE == "REPLAY":
        replay(demo_waypoints.tolist())

    elif MODE == "HEADCAM":
        pose_estimator = PoseEstimation(
            text_prompt=OBJECT,
            demo_rgb_path=f"{DIR}/demo_head_rgb.png",
            demo_depth_path=f"{DIR}/demo_head_depth.png",
            demo_mask_path=f"{DIR}/demo_head_seg.png",
            intrinsics_path="handeye/intrinsics_d415.npy",
            T_WC_path="handeye/T_WC_head.npy"
        )
        
        while True:
            T_delta_world = pose_estimator.run(output_path=f"{DIR}/")
            live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world)
            replay(live_waypoints)

            user_input = input("Continue? (yes/no): ").lower()
            if user_input != 'yes':
                break

            yumi.reset_init()

    elif MODE == "DINOBOT":
        dinobotAlignment = DINOBotAlignment(DIR=DIR)
        error = 1000000
        while error > dinobotAlignment.error_threshold:
            rgb_live_path, depth_live_path = dinobotAlignment.save_rgbd()
            t, R, error = dinobotAlignment.run(rgb_live_path, depth_live_path)
            pose_new_eef_world = dinobotAlignment.compute_new_eef_in_world(R, t, yumi.get_curent_T_left())
            yumi.plan_left_arm(pose_new_eef_world)

        T_bottleneck_left = create_homogeneous_matrix(demo_waypoints[0][:3], demo_waypoints[0][3:])
        T_delta_eef = pose_inv(T_bottleneck_left) @ yumi.get_curent_T_left() 
        live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_eef, reverse=True)
        print(live_waypoints)
        replay(live_waypoints)

    else: 
        raise NotImplementedError

if __name__ == '__main__':
    MODE = sys.argv[1]
    try:
        rospy.init_node('yumi_moveit_demo')
        yumi.init_Moveit()
        file_name=f"{DIR}/demo_bottlenecks.json"

        with open(file_name) as f:
            dbn = json.load(f)
            yumi.reset_init()
            run(dbn, MODE)

        rospy.spin()
    except Exception as e:
        print(f"Error: {e}")