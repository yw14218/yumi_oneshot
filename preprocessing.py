import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

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

def translation_from_matrix(matrix):
    """Extracts the translation vector from a 4x4 homogeneous transformation matrix."""
    return matrix[:3, 3]

def quaternion_from_matrix(matrix):
    """Extracts the quaternion from a 4x4 homogeneous transformation matrix."""
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    return rotation.as_quat()

def apply_transformation(tf_6dof, waypoints):
    """
    Applies a 6-DoF transformation to a list of waypoints.

    Parameters:
    - tf_6dof: A 6-element array-like, first 3 elements are the translation vector, 
      and the last 4 elements are the quaternion (x, y, z, w).
    - waypoints: A list of waypoints, each waypoint is a 7-element array-like,
      first 3 elements are the translation vector, and the last 4 elements are the quaternion.

    Returns:
    - A list of transformed waypoints, in the same format as the input waypoints.
    """
    # Create the transformation matrix from the given 6-DoF transformation
    translation = tf_6dof[:3]
    rotation = R.from_quat(tf_6dof[3:7])
    RT = np.eye(4)
    RT[:3, :3] = rotation.as_matrix()
    RT[:3, 3] = translation

    transformed_waypoints = []

    # Apply the transformation to each waypoint
    for waypoint in waypoints:
        waypoint_translation = waypoint[:3]
        waypoint_rotation = R.from_quat(waypoint[3:7]).as_matrix()

        RW = np.eye(4)
        RW[:3, :3] = waypoint_rotation
        RW[:3, 3] = waypoint_translation

        # Apply the composite transformation
        transformed_matrix = RT @ RW
        transformed_translation = translation_from_matrix(transformed_matrix)
        transformed_quaternion = quaternion_from_matrix(transformed_matrix)

        transformed_waypoints.append(np.concatenate((transformed_translation, transformed_quaternion)).tolist())

    return transformed_waypoints

def apply_transformation_gripper2world(tf_6dof, gripper2world, waypoints):
    """
    Applies a 6-DoF transformation to a list of waypoints.

    Parameters:
    - tf_6dof: A 6-element array-like, first 3 elements are the translation vector, 
      and the last 4 elements are the quaternion (x, y, z, w).
    - waypoints: A list of waypoints, each waypoint is a 7-element array-like,
      first 3 elements are the translation vector, and the last 4 elements are the quaternion.

    Returns:
    - A list of transformed waypoints, in the same format as the input waypoints.
    """
    # Create the transformation matrix from the given 6-DoF transformation
    print(tf_6dof)
    print(gripper2world)
    translation = tf_6dof[:3]
    rotation = R.from_quat(tf_6dof[3:7])
    RT = np.eye(4)
    RT[:3, :3] = rotation.as_matrix()
    RT[:3, 3] = translation

    translation = gripper2world[:3]
    rotation = R.from_quat(gripper2world[3:7])
    gripper2world = np.eye(4)
    gripper2world[:3, :3] = rotation.as_matrix()
    gripper2world[:3, 3] = translation

    transformed_waypoints = []

    # Apply the transformation to each waypoint
    for waypoint in waypoints:
        waypoint_translation = waypoint[:3]
        waypoint_rotation = R.from_quat(waypoint[3:7]).as_matrix()

        RW = np.eye(4)
        RW[:3, :3] = waypoint_rotation
        RW[:3, 3] = waypoint_translation

        # Apply the composite transformation
        transformed_matrix = gripper2world @ RT @ RW
        transformed_translation = translation_from_matrix(transformed_matrix)
        transformed_quaternion = quaternion_from_matrix(transformed_matrix)

        transformed_waypoints.append(np.concatenate((transformed_translation, transformed_quaternion)).tolist())

    return transformed_waypoints
