#!/usr/bin/env python3
from abc import ABC, abstractmethod
import argparse
from experiments import load_experiment

class YuMiExperiment(ABC):
    def __init__(self, dir, object, demo_waypoints, demo_head_rgb, demo_head_mask, demo_wrist_rgb, demo_wrist_mask):
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
        self.demo_head_rgb = demo_head_rgb
        self.demo_head_mask = demo_head_mask
        self.demo_wrist_rgb = demo_wrist_rgb
        self.demo_wrist_mask = demo_wrist_mask

    @abstractmethod
    def replay(self, live_waypoints):
        raise NotImplementedError()

def main(dir):
    experiment = load_experiment(args.dir)
    pose_estimator = PoseEstimation(
        dir=args.dir,
        text_prompt=experiment.object,
        visualize=False)
    
    dinoBotVS = DINOBotVS(dir)

    # Initialize Moveit
    yumi.init_Moveit()
    # yumi.reset_init()

    try:
        while not rospy.is_shutdown():
            
            # Initial head cam alignment
            diff_xyz, diff_rpy = pose_estimator.decouple_run(output_path=f"{dir}/", camera_prefix="d415")

            bottleneck_left_new = experiment.demo_waypoints[0].tolist()
            bottleneck_left_new[0] += diff_xyz[0]
            bottleneck_left_new[1] += diff_xyz[1]
            # bottleneck_left_new[2] += 0.05
            yumi.plan_left_arm(yumi.create_pose(*bottleneck_left_new[:3], *bottleneck_left_new[3:]))

            user_input = input("Proceed with waypoint transformation? (yes/no): ").strip().lower()
            dinoBotVS.run()

    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    import rospy
    from poseEstimation import PoseEstimation
    import moveit_utils.yumi_moveit_utils as yumi
    from dinobot import DINOBotVS

    rospy.init_node('Base Experiment', anonymous=True)
    parser = argparse.ArgumentParser(description='Run Yumi Base Experiment.')
    parser.add_argument('--dir', type=str, help='Directory path for the experiment data.')
    args = parser.parse_args()

    main(args.dir)
    


        
