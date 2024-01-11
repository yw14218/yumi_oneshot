#!/usr/bin/env python3


import sys
import copy
import rospy
import moveit_commander
import yumi_moveit_utils as yumi
import moveit_msgs.msg
import geometry_msgs.msg
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
import json


joint_state_list = []

def joint_state_callback(msg):
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
    joint_state_list.append(joint_state_data)

def write_to_file():
    with open("yumi_joint_states.json", "w") as file:
        json.dump(joint_state_list, file, indent=4)
    rospy.loginfo("Yumi Joint States saved to JSON file")

def listener():
    rospy.init_node('yumi_joint_state_listener', anonymous=True)
    rospy.Subscriber("/yumi/joint_states", JointState, joint_state_callback)
    rospy.on_shutdown(write_to_file)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass