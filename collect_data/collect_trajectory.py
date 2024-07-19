#!/usr/bin/env python3

import sys
import rospy
import json
import argparse
import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Configuration Constant
JOINT_TOPIC_NAME = "/yumi/joint_states"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Yumi Joint State Recorder')
    parser.add_argument('--output_file', type=str, help='Output file for joint states')
    parser.add_argument('--gripper_init_states', type=str, nargs=2, help='Initial gripper states as two strings', default=["open_left", "open_right"])
    args = parser.parse_args()
    return args.output_file, tuple(args.gripper_init_states)

class JointStateHandler:
    def __init__(self, output_file, gripper_init_states):
        self.output_file = output_file
        self.joint_state_list = []
        self.gripper_state_lock = threading.Lock()  # Lock for thread-safe updates to gripper states
        self.latest_gripper_state = gripper_init_states

    def add_joint_state(self, msg):
        with self.gripper_state_lock:  # Ensure thread-safe access to gripper states
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
                "gripper_state": self.latest_gripper_state
            }
        self.joint_state_list.append(joint_state_data)
        print(joint_state_data)

    def update_gripper_state(self, new_state_left, new_state_right):
        with self.gripper_state_lock:
            self.latest_gripper_state = (new_state_left, new_state_right)

    def write_to_file(self):
        try:
            with open(self.output_file, "w") as file:
                json.dump(self.joint_state_list, file, indent=4)
            rospy.loginfo(f"Joint and Gripper States saved to {self.output_file}")
        except IOError as e:
            rospy.logerr(f"Failed to write to file: {e}")

def user_input_thread(handler):
    gripper_states_options = {
        '1': ('open_left', 'open_right'),
        '2': ('close_left', 'close_right'),
        '3': ('open_left', 'close_right'),
        '4': ('close_left', 'open_right'),
    }

    while not rospy.is_shutdown():
        print("\nSelect new gripper states:")
        print("1: Open Left, Open Right")
        print("2: Close Left, Close Right")
        print("3: Open Left, Close Right")
        print("4: Close Left, Open Right")
        print("Type the number of your choice:")

        user_input = input().strip()
        if user_input in gripper_states_options:
            new_state_left, new_state_right = gripper_states_options[user_input]
            handler.update_gripper_state(new_state_left, new_state_right)
            print(f"Gripper states updated to: {new_state_left}, {new_state_right}")
        else:
            print("Invalid selection. Please enter a number corresponding to the options provided.")


class RobotListener:
    def __init__(self, output_file, gripper_init_states):
        self.handler = JointStateHandler(output_file, gripper_init_states)
        rospy.init_node('yumi_joint_gripper_state_listener', anonymous=True)
        rospy.Subscriber(JOINT_TOPIC_NAME, JointState, self.handler.add_joint_state)
        rospy.on_shutdown(self.handler.write_to_file)
        # self.user_input_thread = threading.Thread(target=user_input_thread, args=(self.handler,))
        # self.user_input_thread.start()

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    output_file, gripper_init_states = parse_arguments()
    listener = RobotListener(output_file, gripper_init_states)
    listener.start()


