import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
import matplotlib.pyplot as plt
from pytransform3d.rotations import plot_basis
from pytransform3d.transformations import plot_transform
import matplotlib.animation as animation

# Define the start and end points in 6D (3D position + quaternion)
start_position = np.array([0.49467759, -0.13398115, 0.37963961])
end_position = np.array([0.492955653926487, -0.11312450854488373, 0.28185394586707374])
start_quaternion = np.array([-0.9943035148839005,
        0.01699489712745371,
        0.010448577177212925,
        0.10470205822294058])  # Example quaternion
end_quaternion = np.array([-0.9943035148839005,
        0.01699489712745371,
        0.010448577177212925,
        0.10470205822294058])    # Example quaternion

# Number of waypoints (including start and end)
num_waypoints = 10

# Create a linspace of parameter t
t = np.linspace(0, 1, num_waypoints)

# Linear interpolation for positions
interp_positions = np.array([start_position * (1 - t_i) + end_position * t_i for t_i in t])

# Quaternion slerp interpolation
rotations = Rotation.from_quat([start_quaternion, end_quaternion])
slerp = Slerp([0, 1], rotations)
interp_quaternions = slerp(t).as_quat()

# Combine positions and quaternions
waypoints = np.hstack((interp_positions, interp_quaternions))

# Print the waypoints
print("Waypoints (Position + Quaternion):")
print(waypoints)

# Generate a finer grid for interpolation
t_fine = np.linspace(0, 1, 10)

# Interpolate positions for the finer grid
interp_positions_fine = np.array([start_position * (1 - t_i) + end_position * t_i for t_i in t_fine])

# Quaternion slerp interpolation for the finer grid
interp_quaternions_fine = slerp(t_fine).as_quat()

# Combine positions and quaternions for the finer grid
interpolated_points = np.hstack((interp_positions_fine, interp_quaternions_fine))

# Print the interpolated points
print("\nInterpolated Points (Position + Quaternion):")
print(interpolated_points)

# Visualization using pytransform3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot interpolated positions
ax.plot(interpolated_points[:, 0], interpolated_points[:, 1], interpolated_points[:, 2], label='Interpolated Path', linestyle='--', linewidth=1)

# Plot waypoints
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color='red', label='Waypoints', s=50)

# Plot frames at waypoints
for pos, quat in zip(interp_positions, interp_quaternions):
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
    transform_matrix[:3, 3] = pos
    plot_transform(ax=ax, A2B=transform_matrix, s=0.5)

# Plot frames at finer interpolated points
for pos, quat in zip(interp_positions_fine, interp_quaternions_fine):
    plot_basis(ax=ax, R=Rotation.from_quat(quat).as_matrix(), p=pos, s=0.2)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('6D Waypoint Interpolation (Position + Quaternion)')
ax.legend()
plt.show()

import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_pre_grasp_pose(grasp_pos, grasp_quat, approach_distance):
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(grasp_quat).as_matrix()

    # Approach vector is the negative Z-axis of the end-effector in world frame
    approach_vector = -rotation[:, 2]

    # Compute the pre-grasp position
    pre_grasp_pos = grasp_pos + approach_vector * approach_distance

    # Pre-grasp orientation is the same as grasp orientation
    pre_grasp_quat = grasp_quat

    return pre_grasp_pos, pre_grasp_quat

# Given grasp pose
grasp_pos = np.array([0.492955653926487, -0.11312450854488373, 0.28185394586707374])
grasp_quat = np.array([-0.9943035148839005, 0.01699489712745371, 0.010448577177212925, 0.10470205822294058])

# Approach distance (example value)
approach_distance = 0.1

# Compute pre-grasp pose
pre_grasp_pos, pre_grasp_quat = compute_pre_grasp_pose(grasp_pos, grasp_quat, approach_distance)

# Print results
print("Pre-Grasp Position:", pre_grasp_pos)
print("Pre-Grasp Quaternion:", pre_grasp_quat)
