#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import yaml

class JointStateRecorder:
    def __init__(self):
        self.joint_states = []
        self.joint_names = [
            "yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l",
            "yumi_joint_3_l", "yumi_joint_4_l", "yumi_joint_5_l", "yumi_joint_6_l"
        ]

    def record_joint_state(self):
        msg = rospy.wait_for_message("/joint_states", JointState)
        if all(name in msg.name for name in self.joint_names):
            state = [msg.position[msg.name.index(name)] for name in self.joint_names]
            self.joint_states.append(state)
            formatted_state = "\n".join("  - " + str(value) for value in state)
            print("-\n" + formatted_state)
            print("Current state recorded.")

    def run(self):
        rospy.init_node('joint_state_recorder')
        print("Recording joint states. Enter 'yes' to record current state, 'no' to exit.")
        
        while not rospy.is_shutdown():
            user_input = input("Record current state? (yes/no): ")
            if user_input.lower() == "no":
                break
            elif user_input.lower() == "yes":
                self.record_joint_state()

        self.write_to_yaml()

    def write_to_yaml(self):
        recorded_data = {
            "joint_names": self.joint_names,
            "joint_values": self.joint_states
        }
        with open("recorded_states.yaml", "w") as file:
            yaml.dump(recorded_data, file)
        print("Joint states saved to recorded_states.yaml")

if __name__ == '__main__':
    recorder = JointStateRecorder()
    recorder.run()
