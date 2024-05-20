#!/usr/bin/env python3

import json
import rospy
import numpy as np
import yumi_moveit_utils as yumi
from poseEstimation import PoseEstimation
from trajectory_utils import apply_transformation_to_waypoints, create_homogeneous_matrix, pose_inv, project3D, translation_from_matrix, quaternion_from_matrix
from dinobotAlignment import DINOBotAlignment
from abc import ABC, abstractmethod

class YuMiExperiment(ABC):
    def __init__(self, dir, object, mode):
        self.dir = dir
        self.object = object
        self.mode = mode

    @abstractmethod
    def replay(self, live_waypoints):
        raise NotImplementedError()
    
    def run(self):
        file_name = f"{self.dir}/demo_bottlenecks.json"
        with open(file_name) as f:
            dbn = json.load(f)
        demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

        if self.mode == "REPLAY":
            self.replay(demo_waypoints.tolist())

        elif self.mode == "HEADCAM":
            pose_estimator = PoseEstimation(
                dir=self.dir,
                text_prompt=self.object,
                visualize=False
            )

            while True:
                # T_delta_cam = pose_estimator.run(output_path=f"{self.dir}/", camera_prefix="d415")
                # T_WC = np.load(pose_estimator.T_WC_path)
                # T_delta_world = T_WC @ T_delta_cam @ pose_inv(T_WC)
                T_delta_world = np.eye(4)
                dxyz = pose_estimator.run(output_path=f"{self.dir}/", camera_prefix="d415")
                T_delta_world[0, 3] = dxyz[0]
                T_delta_world[1, 3] = dxyz[1]
                rospy.loginfo("T_delta_world is {0}".format(T_delta_world))
                live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
                yumi.plan_both_arms(live_waypoints[0], live_waypoints[1])
                # self.replay(live_waypoints)
                rospy.sleep(3)

                # raise NotImplementedError

                error = 1000
                while error > 0.005:
                    T_delta_cam = pose_estimator.run(output_path=f"{self.dir}/", camera_prefix="d405")
                    T_camera_eef = np.load(pose_estimator.T_CE_l_path)
                    T_new_eef_world = yumi.get_curent_T_left() @ T_camera_eef @ T_delta_cam @ pose_inv(T_camera_eef)
                    rospy.loginfo("T_delta_world is {0}".format(T_new_eef_world))
                    xyz = translation_from_matrix(T_new_eef_world).tolist()
                    error = np.linalg.norm(T_delta_cam[:3, 3])
                    print(error)
                    quaternion = quaternion_from_matrix(T_new_eef_world).tolist()
                    pose_new_eef_world_l = project3D(xyz + quaternion, demo_waypoints[0])
                    yumi.plan_left_arm(pose_new_eef_world_l)
                    rospy.sleep(0.1)


                    
                    
                # user_input = input("Go? (yes/no): ").lower()
                # if user_input == 'go':
                T_bottleneck_left = create_homogeneous_matrix(demo_waypoints[0][:3], demo_waypoints[0][3:])
                T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
                print(T_delta_world)
                live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
                self.replay(live_waypoints)

                # del pose_estimator
                # dinobot_alignment = DINOBotAlignment(DIR=self.dir)
                # T_bottleneck_left = create_homogeneous_matrix(demo_waypoints[0][:3], demo_waypoints[0][3:])
                # error = 1000000
                
                # while error > dinobot_alignment.error_threshold:
                #     rgb_live_path, depth_live_path = dinobot_alignment.save_rgbd()
                #     t, R, error = dinobot_alignment.run(rgb_live_path, depth_live_path)
                #     current_T_left = yumi.get_curent_T_left()
                #     pose_new_eef_world_l = dinobot_alignment.compute_new_eef_in_world(R, t, current_T_left)
                #     pose_new_eef_world_l = project3D(pose_new_eef_world_l, demo_waypoints[0])
                #     yumi.plan_left_arm(pose_new_eef_world_l)

                # T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
                # live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
                # self.replay(live_waypoints)

                user_input = input("Continue? (yes/no): ").lower()
                if user_input != 'yes':
                    break

                yumi.reset_init()

        elif self.mode == "DINOBOT":
            dinobot_alignment = DINOBotAlignment(DIR=self.dir)
            T_bottleneck_left = create_homogeneous_matrix(demo_waypoints[0][:3], demo_waypoints[0][3:])
            T_bottleneck_right = create_homogeneous_matrix(demo_waypoints[1][:3], demo_waypoints[1][3:])
            T_right_left = pose_inv(T_bottleneck_left) @ T_bottleneck_right 
            error = 1000000
            
            while error > dinobot_alignment.error_threshold:
                rgb_live_path, depth_live_path = dinobot_alignment.save_rgbd()
                t, R, error = dinobot_alignment.run(rgb_live_path, depth_live_path)
                current_T_left = yumi.get_curent_T_left()
                pose_new_eef_world_l = dinobot_alignment.compute_new_eef_in_world(R, t, current_T_left)
                pose_new_eef_world_l = project3D(pose_new_eef_world_l, demo_waypoints[0])
                yumi.plan_left_arm(pose_new_eef_world_l)

                # T_pose_new_eef_world_r = create_homogeneous_matrix(pose_new_eef_world_l[:3], pose_new_eef_world_l[3:]) @ T_right_left
                # pose_new_eef_world_r = translation_from_matrix(T_pose_new_eef_world_r).tolist() + quaternion_from_matrix(T_pose_new_eef_world_r).tolist()
                # yumi.plan_both_arms(pose_new_eef_world_l, pose_new_eef_world_r)

            # T_delta_eef = pose_inv(T_bottleneck_left) @ yumi.get_curent_T_left()
            T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
            live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
            self.replay(live_waypoints)

        else: 
            raise NotImplementedError(f"Mode {self.mode} not implemented")



