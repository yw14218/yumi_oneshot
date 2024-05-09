import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from moveit_commander import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def filter_joint_states(joint_states, threshold):
    """
    Filtering joint states by keeping data points where at least one joint's position changes
    by a value equal to or greater than the specified threshold from its previous state.

    :param joint_states: List of dictionary containing the joint states of the robot arm.
    :param threshold: The minimum change in any joint state required to keep the data point.
    :return: List of dictionaries containing the smoothed trajectory data.
    """
    if not joint_states:
        return []

    # Initialize with the first joint state
    filtered_joint_states = [joint_states[0]]

    for current_state in joint_states[1:]:
        # Compare current state with the last filtered state
        if any(abs(curr - prev) >= threshold for curr, prev in zip(current_state['position'], filtered_joint_states[-1]['position'])):
            filtered_joint_states.append(current_state)

    return filtered_joint_states

# Function to plot joint trajectories
def plot_joint_trajectories(joint_trajectories, title):
    plt.figure(figsize=(12, 8))
    for i, joint_trajectory in enumerate(joint_trajectories):
        # Adjust the label index based on the step size
        plt.plot(joint_trajectory, label=f'Joint {i+1}')

    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Joint States')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_homogeneous_matrix(xyz, quaternion):
    # Convert the quaternion to a rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    # Create a homogeneous transformation matrix
    T = np.eye(4)  # Start with an identity matrix
    T[:3, :3] = rotation_matrix  # Insert the rotation matrix
    T[:3, 3] = xyz  # Insert the translation vector

    return T
    
def translation_from_matrix(matrix):
    """Extracts the translation vector from a 4x4 homogeneous transformation matrix."""
    return matrix[:3, 3]

def quaternion_from_matrix(matrix):
    """Extracts the quaternion from a 4x4 homogeneous transformation matrix."""
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    return rotation.as_quat()

def pose_inv(pose):
    """Inverse a 4x4 homogeneous transformation matrix."""
    R = pose[:3, :3]
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = - R.T @ np.ascontiguousarray(pose[:3, 3])
    return T

def apply_transformation_to_waypoints(waypoints_np, delta_R, reverse=False):
    """
    Apply a transformation to a list of end-effector poses using NumPy vectorization.
    """

    # Separate translations and rotations
    translations = waypoints_np[:, :3]
    rotations = waypoints_np[:, 3:]

    # Convert quaternions to rotation matrices
    waypoint_rot_matrices = R.from_quat(rotations).as_matrix()

    # Construct 4x4 matrices for waypoints
    RW_matrices = np.zeros((waypoints_np.shape[0], 4, 4))
    RW_matrices[:, :3, :3] = waypoint_rot_matrices
    RW_matrices[:, :3, 3] = translations
    RW_matrices[:, 3, 3] = 1

    # Apply the transformation
    transformed_matrices =  RW_matrices @ delta_R if reverse else delta_R @ RW_matrices

    # Extract translations and rotations from transformed matrices
    transformed_translations = transformed_matrices[:, :3, 3]
    transformed_rotations = R.from_matrix(transformed_matrices[:, :3, :3]).as_quat()

    # Concatenate the results
    transformed_waypoints = np.hstack((transformed_translations, transformed_rotations))

    return transformed_waypoints.tolist()

def align_trajectories(plan1, plan2):
    """
    Aligns the durations of two trajectories to have the same number of points by stretching 
    the timeline of the shorter trajectory.

    This function modifies the shorter trajectory by increasing its time_from_start for each point,
    distributing the points evenly over the duration of the longer trajectory. It ensures that both
    trajectories will have points that correspond in time, facilitating synchronization of execution.

    Parameters:
    - plan1 (RobotTrajectory): The first trajectory plan, assumed to be equal to or longer than plan2.
    - plan2 (RobotTrajectory): The second trajectory plan, assumed to be shorter or equal to plan1.

    Returns:
    - None: The function modifies plan2 in place; it does not return a value.

    Note:
    If plan2 is longer than plan1, the function will recursively call itself with swapped arguments.

    Example:
    - If plan1 has 10 points spread over 10 seconds and plan2 has 5 points spread over 5 seconds,
      plan2's points will be adjusted to be 2 seconds apart, matching the 10-second span.

    Raises:
    - This function assumes that both plan1 and plan2 are valid RobotTrajectory objects and that
      they have been properly initialized with joint_trajectory data. It will not work correctly
      if these conditions are not met.
    """

    # Assuming plan1 is longer or equal to plan2
    if len(plan1.joint_trajectory.points) > len(plan2.joint_trajectory.points):
        scale_factor = len(plan1.joint_trajectory.points) / len(plan2.joint_trajectory.points)
        new_points = []
        for point in plan2.joint_trajectory.points:
            new_point = copy.deepcopy(point)
            new_point.time_from_start *= scale_factor
            new_points.append(new_point)
        plan2.joint_trajectory.points = new_points
    elif len(plan1.joint_trajectory.points) < len(plan2.joint_trajectory.points):
        align_trajectories(plan2, plan1)  # Swap roles


def merge_trajectories(plan_left, plan_right):
    """
    Merges two robotic arm trajectory plans into a single plan by combining their joint trajectory points.

    This function assumes that both input plans are already aligned in terms of time and length,
    meaning they have the same number of points and corresponding points occur at the same times.
    It concatenates the joint names and merges the trajectory points from both plans into a single
    trajectory, which includes positions, velocities, accelerations, and efforts.

    Parameters:
    - plan_left (RobotTrajectory): The trajectory plan for the left arm.
    - plan_right (RobotTrajectory): The trajectory plan for the right arm.

    Returns:
    - RobotTrajectory: A new RobotTrajectory object containing the merged trajectory of both input plans.

    Raises:
    - AssertionError: If the lengths of the trajectory points of the two plans do not match.
    
    Example Usage:
    Assume plan_left and plan_right are precomputed RobotTrajectory objects with aligned trajectory points:
    merged_plan = merge_trajectories(plan_left, plan_right)
    """
        
    # Create a new trajectory
    merged_trajectory = RobotTrajectory()
    merged_trajectory.joint_trajectory = JointTrajectory()

    # Combine joint names
    merged_trajectory.joint_trajectory.joint_names = (
        plan_left.joint_trajectory.joint_names + plan_right.joint_trajectory.joint_names
    )

    # Assuming both plans are now aligned in time and length
    for p1, p2 in zip(plan_left.joint_trajectory.points, plan_right.joint_trajectory.points):
        new_point = JointTrajectoryPoint()
        new_point.positions = p1.positions + p2.positions
        new_point.velocities = p1.velocities + p2.velocities if p1.velocities and p2.velocities else []
        new_point.accelerations = p1.accelerations + p2.accelerations if p1.accelerations and p2.accelerations else []
        new_point.effort = p1.effort + p2.effort if p1.effort and p2.effort else []
        new_point.time_from_start = p1.time_from_start
        merged_trajectory.joint_trajectory.points.append(new_point)

    return merged_trajectory