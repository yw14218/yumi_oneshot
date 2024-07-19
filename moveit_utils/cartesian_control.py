import rospy
from ikSolver import IKSolver
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class YuMiCartesianController:
    def __init__(self, group_name="left_arm", ik_link_name="gripper_l_base", joint_names=None, queue_size=10):
        """
        Initialize the CartesianController with an IK solver and a ROS publisher.

        Parameters:
        - group_name: The name of the arm group (default is "left_arm").
        - ik_link_name: The name of the link for the inverse kinematics (default is "gripper_l_base").
        - joint_names: List of joint names for the robot arm (optional).
        - queue_size: Size of the queue for the ROS publisher (default is 10).
        """
        # Initialize the ROS publisher for sending joint trajectory commands
        self.publisher = rospy.Publisher("/yumi/joint_traj_pos_controller_l/command", JointTrajectory, queue_size=queue_size)
        
        # Initialize the IK solver
        self.ik_solver = IKSolver(group_name=group_name, ik_link_name=ik_link_name)
        
        # Default joint names if not provided
        if joint_names is None:
            self.joint_names = [f"yumi_joint_{i}_l" for i in [1, 2, 7, 3, 4, 5, 6]]
        else:
            self.joint_names = joint_names
    
    def move_eef(self, eef_pose):
        """
        Move the end-effector to the desired pose using inverse kinematics.

        Parameters:
        - eef_pose: The desired pose of the end-effector (usually a geometry_msgs/Pose or equivalent).
        """
        # Compute the inverse kinematics to get joint positions
        ik_result = self.ik_solver.get_ik(eef_pose).solution.joint_state.position
        target_joints = ik_result[:7]
        
        # Create and populate the JointTrajectory message
        msg = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = rospy.Duration(0.05)
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self.joint_names
        msg.points.append(point)
        
        # Publish the JointTrajectory message to move the robot
        self.publisher.publish(msg)