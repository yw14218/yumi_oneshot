import rospy
import ros_numpy
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image
import os
import json
import yumi_moveit_utils as yumi
import numpy as np
from geometry_msgs.msg import Pose

def dict_to_ros_pose(pose_dict):
    """
    Convert a dictionary representing robot poses into ROS Pose objects.

    Args:
    - pose_dict: A dictionary containing the pose data.

    Returns:
    A dictionary with keys corresponding to those in the input, each mapping to a ROS Pose object.
    """
    ros_poses = {}

    for key in pose_dict.keys():  # Iterate over keys ('start', 'end', etc.)
        pose_data = pose_dict[key]
        pose = Pose()

        # Set position
        pose.position.x = pose_data['position']['x']
        pose.position.y = pose_data['position']['y']
        pose.position.z = pose_data['position']['z']

        # Set orientation
        pose.orientation.x = pose_data['orientation']['x']
        pose.orientation.y = pose_data['orientation']['y']
        pose.orientation.z = pose_data['orientation']['z']
        pose.orientation.w = pose_data['orientation']['w']

        ros_poses[key] = pose

    return ros_poses


def init_node():
    rospy.init_node('collect_demo_left', anonymous=True)

def wait_for_images(rgb_topic, depth_topic):
    rgb_message = rospy.wait_for_message(rgb_topic, ImageMsg)
    depth_message = rospy.wait_for_message(depth_topic, ImageMsg)
    rgb_data = ros_numpy.numpify(rgb_message)
    depth_data = ros_numpy.numpify(depth_message)
    rgb_image = Image.fromarray(rgb_data)
    depth_image = Image.fromarray(depth_data)
    return rgb_image, depth_image

def save_images(rgb_image, depth_image, save_path, prefix):
    rgb_image.save(os.path.join(save_path, f"{prefix}_rgb.png"))
    depth_image.save(os.path.join(save_path, f"{prefix}_depth.png"))

def pose_to_dict(pose):
    return {
        'position': {
            'x': pose.position.x,
            'y': pose.position.y,
            'z': pose.position.z
        },
        'orientation': {
            'x': pose.orientation.x,
            'y': pose.orientation.y,
            'z': pose.orientation.z,
            'w': pose.orientation.w
        }
    }

def get_and_save_pose(yumi_obj, poses_dict, key):
    cur_pose = yumi_obj.get_current_pose(yumi_obj.LEFT).pose
    poses_dict[key] = pose_to_dict(cur_pose)

def save_poses(poses_dict, save_path):
    with open(os.path.join(save_path, "left_demo_poses.json"), 'w') as f:
        json.dump(poses_dict, f, indent=4)

def main():
    init_node()
    save_dir = "data/lego_split"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    yumi.init_Moveit()
    left_demo_poses = {}

    # Collect and save initial images and pose
    rgb_image, depth_image = wait_for_images("camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw")
    rgb_image.show()
    save_images(rgb_image, depth_image, save_dir, "left_demo_start")
    get_and_save_pose(yumi, left_demo_poses, "start")

    input("Please move the robot, when finished, press Enter.")

    # Collect and save final images and pose
    rgb_image, depth_image = wait_for_images("camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw")
    rgb_image.show()
    save_images(rgb_image, depth_image, save_dir, "left_demo_end")
    get_and_save_pose(yumi, left_demo_poses, "end")

    save_poses(left_demo_poses, save_dir)

    print(f"Data saved in {save_dir}")

if __name__ == '__main__':
    try:
       main()
    except Exception as e:
        print(f"Error: {e}")


