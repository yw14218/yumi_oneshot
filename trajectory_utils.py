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

def apply_transformation_to_waypoints(waypoints_np, delta_R, project3D=False):
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
    transformed_matrices =  delta_R @ RW_matrices

    # Extract translations and rotations from transformed matrices
    transformed_translations = transformed_matrices[:, :3, 3]
    transformed_rotations = R.from_matrix(transformed_matrices[:, :3, :3]).as_quat()

    if project3D:
        rpy = R.from_matrix(delta_R[:3, :3]).as_euler("xyz")
        yaw_only_delta_rotation = R.from_euler("xyz", [0, 0, rpy[-1]]).as_matrix()
        yaw_only_transformed_rotations = yaw_only_delta_rotation @ waypoint_rot_matrices
        transformed_translations[:, 2] = translations[:, 2]
        transformed_rotations = R.from_matrix(yaw_only_transformed_rotations).as_quat()


    # Concatenate the results
    transformed_waypoints = np.hstack((transformed_translations, transformed_rotations))

    return transformed_waypoints.tolist()

def project3D(new_pose, ori_pose):
    project_pose = copy.deepcopy(new_pose)
    project_pose[2] = ori_pose[2]

    ori_euler = R.from_quat(ori_pose[3:]).as_euler("xyz")
    new_euler = R.from_quat(new_pose[3:]).as_euler("xyz")

    new_euler[0] = ori_euler[0]
    new_euler[1] = ori_euler[1]

    project_pose[3:] = R.from_euler("xyz", new_euler).as_quat()
    return project_pose

def align_trajectory_points(plan_left, plan_right):
    """
    Aligns the number of points in two trajectory plans by interpolating or decimating points as needed.

    Parameters:
    - plan_left (RobotTrajectory): The trajectory plan for the left arm.
    - plan_right (RobotTrajectory): The trajectory plan for the right arm.

    Returns:
    - Tuple[RobotTrajectory, RobotTrajectory]: A tuple containing the aligned trajectory plans for both arms.
    """
    # Get the number of points in each trajectory
    num_points_left = len(plan_left.joint_trajectory.points)
    num_points_right = len(plan_right.joint_trajectory.points)

    # Calculate the ratio of points between the two trajectories
    ratio = num_points_left / num_points_right

    # Determine which plan has fewer points and interpolate or decimate points in the other plan
    if num_points_left < num_points_right:
        new_points = []
        for i in range(num_points_left):
            idx = int(np.round(i / ratio))
            new_points.append(plan_right.joint_trajectory.points[idx])
        plan_right.joint_trajectory.points = new_points
    elif num_points_left > num_points_right:
        new_points = []
        for i in range(num_points_right):
            idx = int(np.round(i * ratio))
            new_points.append(plan_left.joint_trajectory.points[idx])
        plan_left.joint_trajectory.points = new_points

    return plan_left, plan_right


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

    assert len(plan_left.joint_trajectory.points) == len(plan_right.joint_trajectory.points), "lengths of the trajectory points of the two plans do not match"
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

def compute_pre_grasp_pose(grasp_pos, grasp_quat, approach_distance=0.1):
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(grasp_quat).as_matrix()

    # Approach vector is the negative Z-axis of the end-effector in world frame
    approach_vector = -rotation[:, 2]

    # Compute the pre-grasp position
    pre_grasp_pos = grasp_pos + approach_vector * approach_distance

    # Pre-grasp orientation is the same as grasp orientation
    pre_grasp_quat = grasp_quat

    return np.concatenate([pre_grasp_pos, pre_grasp_quat])