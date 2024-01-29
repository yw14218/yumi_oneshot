#!/usr/bin/env python3

import sys
import rospy
import json
import argparse
from sensor_msgs.msg import JointState

# Configuration Constant
TOPIC_NAME = "/yumi/joint_states"

class JointStateHandler:
    def __init__(self, output_file):
        self.output_file = output_file
        self.joint_state_list = []

    def add_joint_state(self, msg):
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
            "effort": msg.effort
        }
        self.joint_state_list.append(joint_state_data)

    def write_to_file(self):
        try:
            with open(self.output_file, "w") as file:
                json.dump(self.joint_state_list, file, indent=4)
            rospy.loginfo(f"Joint States saved to {self.output_file}")
        except IOError as e:
            rospy.logerr(f"Failed to write to file: {e}")

class RobotListener:
    def __init__(self, topic_name, output_file):
        self.handler = JointStateHandler(output_file)
        rospy.init_node('yumi_joint_state_listener', anonymous=True)
        rospy.Subscriber(topic_name, JointState, self.handler.add_joint_state)
        rospy.on_shutdown(self.handler.write_to_file)

    def start(self):
        rospy.spin()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Yumi Joint State Recorder')
    parser.add_argument('output_file', type=str, help='Output file for joint states')
    args = parser.parse_args()
    return args.output_file

if __name__ == '__main__':
    try:
        output_file = parse_arguments()
        listener = RobotListener(TOPIC_NAME, output_file)
        listener.start()
    except rospy.ROSInterruptException as e:
        rospy.logerr(f"ROS Interrupted: {e}")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")

