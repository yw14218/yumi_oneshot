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


def apply_transformation_to_waypoints(eef_poses, delta_R):
    """
    Apply a transformation to a list of end-effector poses using NumPy vectorization.

    Parameters:
    eef_poses (list): List of end-effector ROS pose_stamped.
    delta_R (numpy.ndarray): 4x4 transformation matrix.

    Returns:
    list: Transformed waypoints.
    """
    # Extract waypoint data
    waypoints = [eef_pose.pose_stamped[0].pose for eef_pose in eef_poses]
    waypoints_np= np.array([[waypoint.position.x, waypoint.position.y, waypoint.position.z,
                               waypoint.orientation.x, waypoint.orientation.y, waypoint.orientation.z,
                               waypoint.orientation.w] for waypoint in waypoints])

    # Separate translations and rotations
    translations = waypoints_np[:, :3]
    rotations = waypoints_np[:, 3:]

    # Convert quaternions to rotation matrices
    waypoint_rot_matrices = R.from_quat(rotations).as_matrix()

    # Construct 4x4 matrices for waypoints
    RW_matrices = np.zeros((len(waypoints), 4, 4))
    RW_matrices[:, :3, :3] = waypoint_rot_matrices
    RW_matrices[:, :3, 3] = translations
    RW_matrices[:, 3, 3] = 1

    # Apply the transformation
    transformed_matrices = delta_R @ RW_matrices

    # Extract translations and rotations from transformed matrices
    transformed_translations = transformed_matrices[:, :3, 3]
    transformed_rotations = R.from_matrix(transformed_matrices[:, :3, :3]).as_quat()

    # Concatenate the results
    transformed_waypoints = np.hstack((transformed_translations, transformed_rotations))

    return transformed_waypoints.tolist()