import rospy
import json
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from trajectory_utils import filter_joint_states
from std_msgs.msg import Float64

class JointStatePublisher:
    def __init__(self, input_file):
        self.input_file = input_file
        
        # Topic names for left and right arm controllers
        self.topic_name_left = "/yumi/joint_traj_pos_controller_l/command"
        self.topic_name_right = "/yumi/joint_traj_pos_controller_r/command"

        # Gripper effort command topics
        self.gripper_left_topic = "/yumi/gripper_l_effort_cmd"
        self.gripper_right_topic = "/yumi/gripper_r_effort_cmd"

        # Read and filter joint states from file
        self.joint_states = filter_joint_states(self.read_from_file(), 0.01)
        
        # Initialize ROS publishers for each arm
        self.publisher_left = rospy.Publisher(self.topic_name_left, JointTrajectory, queue_size=10)
        self.publisher_right = rospy.Publisher(self.topic_name_right, JointTrajectory, queue_size=10)
        
        # Initialize publishers for grippers
        self.gripper_left_publisher = rospy.Publisher(self.gripper_left_topic, Float64, queue_size=10)
        self.gripper_right_publisher = rospy.Publisher(self.gripper_right_topic, Float64, queue_size=10)

    def read_from_file(self):
        try:
            with open(self.input_file, "r") as file:
                return json.load(file)
        except IOError as e:
            rospy.logerr(f"Failed to read from file: {e}")
            return []

    def publish_joint_states(self):
        rospy.init_node('yumi_joint_state_publisher', anonymous=True)
        rate = rospy.Rate(25)  # Define loop rate (10 Hz)

        past_gripper_states = [None, None]
        while not rospy.is_shutdown():
            for i, joint_state in enumerate(self.joint_states):
                # Publish for left arm
                joint_traj_left = self.create_joint_trajectory(joint_state, '_l')
                self.publisher_left.publish(joint_traj_left)

                # Publish for right arm
                joint_traj_right = self.create_joint_trajectory(joint_state, '_r')
                self.publisher_right.publish(joint_traj_right)

                current_gripper_states = joint_state["gripper_state"]

                # Publish gripper efforts based on current states
                # It's critical to ensure that current_gripper_states list is structured correctly,
                # i.e., [state_for_left_gripper, state_for_right_gripper]
                self.gripper_left_publisher.publish(-20.0 if current_gripper_states[0] == 'open_left' else 20.0)
                self.gripper_right_publisher.publish(-20.0 if current_gripper_states[1] == 'open_right' else 20.0)

                # Check if there's a change in gripper states compared to the past states
                if past_gripper_states[0] != current_gripper_states[0] or past_gripper_states[1] != current_gripper_states[1]:
                    rospy.sleep(4)  # Delay if gripper states have changed

                # if i == 100:
                #     rospy.sleep(2)

                # Update past_gripper_states for the next iteration
                past_gripper_states = current_gripper_states
                rospy.loginfo(f"Published joint state {i}")
                rospy.loginfo(current_gripper_states)
                rate.sleep()
            break

    def create_joint_trajectory(self, joint_state, arm_suffix):
        """Helper method to create a joint trajectory for a given arm."""
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.joint_names = [name for name in joint_state["name"] if arm_suffix in name]
        
        point = JointTrajectoryPoint()
        point.positions = [joint_state["position"][i] for i, name in enumerate(joint_state["name"]) if arm_suffix in name]
        point.effort = [joint_state["effort"][i] for i, name in enumerate(joint_state["name"]) if arm_suffix in name]
        point.time_from_start = rospy.Duration(0.2)
        
        joint_traj.points = [point]
        return joint_traj

if __name__ == '__main__':
    input_file = "split_lego_both.json"  # Ensure correct path
    joint_state_publisher = JointStatePublisher(input_file)
    joint_state_publisher.publish_joint_states()
