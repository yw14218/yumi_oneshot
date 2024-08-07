#!/usr/bin/env python3
from abc import ABC, abstractmethod
import argparse
from experiments import load_experiment
from trajectory_utils import euler_from_matrix
import numpy as np

class YuMiExperiment(ABC):
    def __init__(self, dir, object, demo_waypoints, rearrange_waypoints, demo_head_rgb, demo_head_mask, demo_wrist_rgb, demo_wrist_mask):
        """
        Initialize an ExperimentData instance with the provided data.

        :param dir: Directory path containing the experiment data.
        :param object: Description or label for the experiment object.
        :param demo_waypoints: Array of waypoints for the experiment.
        :param demo_head_rgb: RGB image from the head camera.
        :param demo_head_mask: Mask image from the head camera.
        :param demo_wrist_rgb: RGB image from the wrist camera.
        :param demo_wrist_mask: Mask image from the wrist camera.
        """
        self.dir = dir
        self.object = object
        self.demo_waypoints = demo_waypoints
        self.rearrange_waypoints=rearrange_waypoints
        self.demo_head_rgb = demo_head_rgb
        self.demo_head_mask = demo_head_mask
        self.demo_wrist_rgb = demo_wrist_rgb
        self.demo_wrist_mask = demo_wrist_mask

    @abstractmethod
    def replay(self, live_waypoints):
        raise NotImplementedError()
    
    @abstractmethod
    def rearrange(self, rearrange_pose, arm):
        raise NotImplementedError()

def main(dir):
    experiment = load_experiment(args.dir)
    # pose_estimator = PoseEstimation(
    #     dir=args.dir,
    #     text_prompt=experiment.object,
    #     visualize=False)
    
    # dinoBotVS = DINOBotVS(dir)
    relCamVS = _HierachicalVS(dir)
    # homVS = HomVS(dir)

    # Initialize Moveit
    yumi.init_Moveit()
    # yumi.reset_init()

    bottleneck_left = experiment.demo_waypoints[0].tolist()
    T_bottleneck_left = create_homogeneous_matrix(bottleneck_left[:3], bottleneck_left[3:])

    try:
        while not rospy.is_shutdown():
            user_input = input("Proceed with waypoint transformation? (yes/no): ").strip().lower()
            # Initial head cam alignment
            # diff_xyz, diff_rpy = pose_estimator.decouple_run(output_path=f"{dir}/", camera_prefix="d415")
            # bottleneck_left[0] += diff_xyz[0]
            # bottleneck_left[1] += diff_xyz[1]
            # bottleneck_left[2] += 0.2

            yumi.plan_left_arm(yumi.create_pose(*bottleneck_left[:3], *bottleneck_left[3:]))
            # user_input = input("Proceed with waypoint transformation? (yes/no): ").strip().lower()
            # del pose_estimator
            # homVS.run()
            relCamVS.run()
            # T_delta_world =  yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
            # rz = np.rad2deg(euler_from_matrix(T_delta_world)[-1])
            # if abs(rz) > 50:
            #     # Rearrange experiment
            #     demo_rearrange = experiment.rearrange_waypoints
            #     live_rearrange = apply_transformation_to_waypoints(demo_rearrange, T_delta_world, project3D=True)[0]
            #     experiment.rearrange(live_rearrange, demo_rearrange[0].tolist(), yumi.LEFT)
            #     yumi.plan_left_arm(yumi.create_pose(*bottleneck_left[:3], *bottleneck_left[3:]))

            # # Replay experiment
            # homVS.run()
            # T_delta_world =  yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
            # live_waypoints = apply_transformation_to_waypoints(experiment.demo_waypoints, T_delta_world, project3D=True)
            # experiment.replay(live_waypoints)ea
            break

    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    import rospy
    from poseEstimation import PoseEstimation
    import moveit_utils.yumi_moveit_utils as yumi
    from dinobot import DINOBotVS
    from relative_cam_vs import RelCamVS, RelCamVSDust3R
    from relative_cam_UKF import _PVBSKF, _HierachicalVS
    from homography_vs import HomVS
    from trajectory_utils import apply_transformation_to_waypoints, create_homogeneous_matrix, pose_inv

    rospy.init_node('Base Experiment', anonymous=True, log_level=rospy.ERROR)
    parser = argparse.ArgumentParser(description='Run Yumi Base Experiment.')
    parser.add_argument('--dir', type=str, help='Directory path for the experiment data.')
    args = parser.parse_args()

    main(args.dir)
    


        
