#!/usr/bin/env python3

import json
import rospy
import numpy as np
import yumi_moveit_utils as yumi
from poseEstimation import PoseEstimation
from trajectory_utils import apply_transformation_to_waypoints, create_homogeneous_matrix, pose_inv
from dinobotAlignment import DINOBotAlignment
from abc import ABC, abstractmethod

class YuMiExperiment(ABC):
    def __init__(self, dir, object, mode, yumi):
        self.dir = dir
        self.object = object
        self.mode = mode
        self.yumi = yumi

    @abstractmethod
    def replay(self, live_waypoints):
        raise NotImplementedError()
    
    def run(self):
        file_name = f"{self.dir}/demo_bottlenecks.json"
        with open(file_name) as f:
            dbn = json.load(f)
            # self.yumi.reset_init()
        demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

        if self.mode == "REPLAY":
            self.replay(demo_waypoints.tolist())

        elif self.mode == "HEADCAM":
            pose_estimator = PoseEstimation(
                text_prompt=self.object,
                demo_rgb_path=f"{self.dir}/demo_head_rgb.png",
                demo_depth_path=f"{self.dir}/demo_head_depth.png",
                demo_mask_path=f"{self.dir}/demo_head_seg.png",
                intrinsics_path="handeye/intrinsics_d415.npy",
                T_WC_path="handeye/T_WC_head.npy"
            )
            
            while True:
                T_delta_world = pose_estimator.run(output_path=f"{self.dir}/")
                live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world)
                self.replay(live_waypoints)

                user_input = input("Continue? (yes/no): ").lower()
                if user_input != 'yes':
                    break

                yumi.reset_init()

        elif self.mode == "DINOBOT":
            dinobot_alignment = DINOBotAlignment(DIR=self.dir)
            error = 1000000
            while error > dinobot_alignment.error_threshold:
                rgb_live_path, depth_live_path = dinobot_alignment.save_rgbd()
                t, R, error = dinobot_alignment.run(rgb_live_path, depth_live_path)
                pose_new_eef_world = dinobot_alignment.compute_new_eef_in_world(R, t, yumi.get_curent_T_left())
                yumi.plan_left_arm(pose_new_eef_world)

            T_bottleneck_left = create_homogeneous_matrix(demo_waypoints[0][:3], demo_waypoints[0][3:])
            T_delta_eef = pose_inv(T_bottleneck_left) @ yumi.get_curent_T_left()
            live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_eef, reverse=True)
            self.replay(live_waypoints)

        else: 
            raise NotImplementedError(f"Mode {self.mode} not implemented")



