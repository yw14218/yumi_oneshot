#!/usr/bin/env python3

import sys
import rospy
import json
import argparse
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Configuration Constant
JOINT_TOPIC_NAME = "/yumi/joint_states"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Yumi Joint State Recorder')
    parser.add_argument('--output_file', type=str, help='Output file for joint states')
    args = parser.parse_args()
    if args.output_file is None:
        print("Output file is required")
        sys.exit(1)
    return args.output_file

class JointStateHandler:
    def __init__(self, output_file):
        self.output_file = output_file
        self.joint_state_list = []

    def add_joint_state(self, msg):
        gripper_status = rospy.get_param('/gripper_status', 'open')
        joint_state_data = {
            "header": {
                "seq": msg.header.seq,
                "stamp": {
                    "secs": msg.header.stamp.secs,
                    "nsecs": msg.header.stamp.nsecs
                },
                "frame_id": msg.header.frame_id
            },
            "name": msg.name,
            "position": msg.position,
            "velocity": msg.velocity,
            "effort": msg.effort,
            "gripper_status": gripper_status 
        }
        self.joint_state_list.append(joint_state_data)
        rospy.loginfo(f"Joint state added: {joint_state_data}")
        rospy.loginfo("Joint state added with gripper status: %s" % gripper_status)

    def write_to_file(self):
        try:
            with open(self.output_file, "w") as file:
                json.dump(self.joint_state_list, file, indent=4)
            rospy.loginfo(f"Joint States saved to {self.output_file}")
        except IOError as e:
            rospy.logerr(f"Failed to write to file: {e}")

class RobotListener:
    def __init__(self, output_file):
        self.handler = JointStateHandler(output_file)
        rospy.init_node('yumi_joint_state_listener', anonymous=True)
        rospy.Subscriber(JOINT_TOPIC_NAME, JointState, self.handler.add_joint_state)
        rospy.on_shutdown(self.handler.write_to_file)

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    output_file = parse_arguments()
    listener = RobotListener(output_file)
    listener.start()

# rosparam set /gripper_status "open"
