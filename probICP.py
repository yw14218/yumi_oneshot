import rospy
import argparse
from trajectory_utils import pose_inv, apply_transformation_to_waypoints
from poseEstimation import PoseEstimation
from experiments import load_experiment
# import moveit_utils.yumi_moveit_utils as yumi
from camera_utils import d415_T_WC as T_WC, decompose_covariance_matrix

def main(directory):
    rospy.init_node('yumi_prob_ICP', anonymous=True)

    # Load experiment data
    experiment = load_experiment(directory)

    # Initialize pose estimator
    pose_estimator = PoseEstimation(
        dir=directory,
        text_prompt=experiment.object,
        visualize=True
    )

    # # Initialize Moveit
    # yumi.init_Moveit()
    # yumi.reset_init()

    try:
        while not rospy.is_shutdown():
            # Initial head cam alignment
            T_delta_cam = pose_estimator.run(output_path=f"{directory}/", camera_prefix="d415", probICP=False)
            # T_delta_world = T_WC @ T_delta_cam @ pose_inv(T_WC)
            # print(decompose_covariance_matrix(cov_matrix))

            user_input = input("Proceed with waypoint transformation? (yes/no): ").strip().lower()
            # if user_input == "no":
            #     continue 

            # # Transform waypoints
            # live_waypoints = apply_transformation_to_waypoints(experiment.demo_waypoints, T_delta_world, project3D=True)

            # # Replay experiment
            # experiment.replay(live_waypoints)

            # # User input loop
            # user_input = input("Continue? (yes/no/reset): ").strip().lower()
            # if user_input == "yes":
            #     experiment.reset()
            #     if input("Ready? (yes/no): ").strip().lower() != "yes":
            #         break
            # elif user_input == "reset":
            #     experiment.reset()
            #     break
            # else:
            #     break

    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Yumi Prob ICP Experiment.')
    parser.add_argument('--dir', type=str, help='Directory path for the experiment data.')
    args = parser.parse_args()
    main(args.dir)

# 2 3