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
                T_delta_cam = pose_estimator.run(output_path=f"{self.dir}/", camera_prefix="d415")
                T_WC = np.load(pose_estimator.T_WC_path)
                T_delta_world = T_WC @ T_delta_cam @ pose_inv(T_WC)
                # T_delta_world = pose_estimator.run_image_match(output_path=f"{self.dir}/", camera_prefix="d415")
                rospy.loginfo("T_delta_world is {0}".format(T_delta_world))
                live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
                yumi.plan_both_arms(live_waypoints[0], live_waypoints[1])
                # self.replay(live_waypoints)
                rospy.sleep(1)
        
                # error = 1000
                # while error > 0.005:
                #     T_delta_cam = pose_estimator.run(output_path=f"{self.dir}/", camera_prefix="d405")
                #     T_camera_eef = np.load(pose_estimator.T_CE_l_path)
                #     T_new_eef_world = yumi.get_curent_T_left() @ T_camera_eef @ T_delta_cam @ pose_inv(T_camera_eef)
                #     rospy.loginfo("T_delta_world is {0}".format(T_new_eef_world))
                #     xyz = translation_from_matrix(T_new_eef_world).tolist()
                #     error = np.linalg.norm(T_delta_cam[:3, 3])
                #     print(error)
                #     quaternion = quaternion_from_matrix(T_new_eef_world).tolist()
                #     pose_new_eef_world_l = project3D(xyz + quaternion, demo_waypoints[0])
                #     yumi.plan_left_arm(pose_new_eef_world_l)
                #     rospy.sleep(0.1)
     
                # T_bottleneck_left = create_homogeneous_matrix(demo_waypoints[0][:3], demo_waypoints[0][3:])
                # T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
                # live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
                # self.replay(live_waypoints)

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

        elif self.mode == "KEL":
            import torch
            import cv2
            import copy
            from PIL import Image, ImageDraw
            import matplotlib.pyplot as plt
            pose_estimator = PoseEstimation(
                dir=self.dir,
                text_prompt=self.object,
                visualize=False
            )

            while True:
                T_delta_cam = pose_estimator.run(output_path=f"{self.dir}/", camera_prefix="d415")
                T_WC = np.load(pose_estimator.T_WC_path)
                T_delta_world = T_WC @ T_delta_cam @ pose_inv(T_WC)
                rospy.loginfo("T_delta_world is {0}".format(T_delta_world))
                live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
                yumi.plan_both_arms(live_waypoints[0], live_waypoints[1])
                # yumi.plan_left_arm(live_waypoints[0])
                # self.replay(live_waypoints)
                rospy.sleep(1)
        
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

                rgb_image, depth_image, mask_image = pose_estimator.inference_and_save("d405", f"{self.dir}/")
                live_rgb_seg = np.array(rgb_image) * np.array(mask_image).astype(bool)[..., None]
                demo_rgb_seg = np.array(Image.open(f"{self.dir}/demo_wrist_rgb_seg.png"))
                xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)
                mkpts_0, mkpts_1 = xfeat.match_xfeat_star(demo_rgb_seg, live_rgb_seg, top_k = 4096)
                H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 5.0)
                stable_point = [499.86090324, 103.8931002]
                key_point_img1 = np.array([499.86090324, 103.8931002], dtype=np.float32).reshape(-1, 1, 2)
                key_point_img2_hom = cv2.perspectiveTransform(key_point_img1, H)
                x_prime, y_prime = key_point_img2_hom[0][0]
                rgb_array = np.array(rgb_image)
                x, y, r = x_prime, y_prime, 5
                fig, ax = plt.subplots()
                ax.imshow(rgb_array)
                circle = plt.Circle((x, y), r, edgecolor='red', linewidth=3, fill=False)
                ax.add_patch(circle)
                ax.axis('off')
                plt.show()
                    
                K = np.load(pose_estimator.intrinsics_d405_path)
                T_C_EEF = np.load(pose_estimator.T_CE_l_path)
                z = np.array(depth_image)[int(y_prime), int(x_prime)]

                def convert_from_uvd(K, u, v, d):
                    fx, fy = K[0, 0], K[1, 1]
                    cx, cy = K[0, 2], K[1, 2]
                    x_over_z = (u - cx) / fx
                    y_over_z = (v - cy) / fy
                    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
                    x = x_over_z * z
                    y = y_over_z * z
                    
                    return x, y, z

                x, y, z = convert_from_uvd(K, x_prime, y_prime, z /1000)
                T_EEF_WORLD = yumi.get_curent_T_left() 
                T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.136], [0, 0, 0, 1])
                yumi.static_tf_broadcast("d405_color_optical_frame", "goal_camera", [x, y, z, 0, 0, 0, 1])

                xyz_world = (T_EEF_WORLD @ T_C_EEF @ create_homogeneous_matrix([x, y, z], [0, 0, 0, 1]))[:3, 3]
                yumi.static_tf_broadcast("world", "goal_world", [xyz_world[0],xyz_world[1], xyz_world[2], 0, 0, 0, 1])

                goal_pose = copy.deepcopy(demo_waypoints[3])

                T_grip_world = create_homogeneous_matrix([xyz_world[0], xyz_world[1], xyz_world[2]], goal_pose[3:])
                T_eef_world = T_grip_world @ pose_inv(T_GRIP_EEF)
                xyz_eef_world = T_eef_world[:3, 3].tolist()
                q_eef_world = quaternion_from_matrix(T_eef_world).tolist()
                yumi.static_tf_broadcast("world", "eef_world", xyz_eef_world+q_eef_world)
                goal_pose[0] = xyz_eef_world[0] 
                goal_pose[1] = xyz_eef_world[1] 

                yumi.reset_init(yumi.LEFT)
                yumi.close_grippers(yumi.RIGHT)
                yumi.plan_right_arm(yumi.create_pose(*goal_pose))

                # T_bottleneck_left = create_homogeneous_matrix(demo_waypoints[0][:3], demo_waypoints[0][3:])
                # T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
                # live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
                # self.replay(live_waypoints)

                user_input = input("Continue? (yes/no): ").lower()
                if user_input != 'yes':
                    break

                yumi.reset_init()

        else: 
            raise NotImplementedError(f"Mode {self.mode} not implemented")



