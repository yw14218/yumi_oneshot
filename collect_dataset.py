from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ikSolver import IKSolver
import rospy
import ros_numpy
import matplotlib.pyplot as plt
import numpy as np
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image
from trajectory_utils import euler_from_quat
import json
import yumi_moveit_utils as yumi
from zipfile import ZipFile
import os

rospy.init_node('collect_images', anonymous=True)
yumi.init_Moveit()

publisher_left = rospy.Publisher("/yumi/joint_traj_pos_controller_l/command", JointTrajectory, queue_size=10)
ik_solver_left = IKSolver(group_name="left_arm", ik_link_name="gripper_l_base")

file_name = f"experiments/pencile_sharpener/demo_bottlenecks.json"
with open(file_name) as f:
    dbn = json.load(f)
demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

bottleneck_left = demo_waypoints[0].tolist()

# current_rpy = euler_from_quat([bottleneck_left[3], bottleneck_left[4], bottleneck_left[5], bottleneck_left[6]])
# yumi.plan_left_arm(yumi.create_pose_euler(bottleneck_left[0], bottleneck_left[1], bottleneck_left[2], current_rpy[0], current_rpy[1], current_rpy[2]))

DIR = "dataset/scoop"

unit = 0.03 * 5
bottleneck_left[0] += unit
bottleneck_left[1] += unit
bottleneck_left[2] += unit 
for j in range(5):
    current_rpy = euler_from_quat([bottleneck_left[3], bottleneck_left[4], bottleneck_left[5], bottleneck_left[6]])
    current_rpy[-1] += np.radians(15 * (j+1))
    yumi.plan_left_arm(yumi.create_pose_euler(bottleneck_left[0], bottleneck_left[1], bottleneck_left[2], current_rpy[0], current_rpy[1], current_rpy[2]))

    for i in range(50):
        rgb_message_wrist = rospy.wait_for_message("d405/color/image_rect_raw", ImageMsg)
        depth_message_wrist = rospy.wait_for_message("d405/aligned_depth_to_color/image_raw", ImageMsg)
        rgb_data_wrist = ros_numpy.numpify(rgb_message_wrist)
        depth_data_wrist = ros_numpy.numpify(depth_message_wrist)
        rgb_image_wrist = Image.fromarray(rgb_data_wrist)
        depth_image_wrist = Image.fromarray(depth_data_wrist)
        rgb_image_wrist.save(f"{DIR}/wrist_rgb_t{unit}_rot{15*(j+1)}_{i+1}.png")
        depth_image_wrist.save(f"{DIR}/wrist_depth_t{unit}_rot{15*(j+1)}_{i+1}.png")

    # Create a zip file and add the images
    zip_filename = f"{DIR}/t{unit}_rot{15*(j+1)}.zip"
    with ZipFile(zip_filename, 'w') as zipf:
        for i in range(50):
            rgb_filename = f"{DIR}/wrist_rgb_t{unit}_rot{15*(j+1)}_{i+1}.png"
            depth_filename = f"{DIR}/wrist_depth_t{unit}_rot{15*(j+1)}_{i+1}.png"
            zipf.write(rgb_filename)
            zipf.write(depth_filename)

    # Remove the images after adding them to the zip file
    for i in range(50):
        os.remove(f"{DIR}/wrist_rgb_t{unit}_rot{15*(j+1)}_{i+1}.png")
        os.remove(f"{DIR}/wrist_depth_t{unit}_rot{15*(j+1)}_{i+1}.png")

    print(f"Zip file {zip_filename} created successfully.")

    rospy.sleep(3)