import numpy as np
import json
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from movement_primitives.kinematics import Kinematics
from trajectory_utils import filter_joint_states

def animation_callback(step, graph, chain, joint_trajectory):
    chain.forward(joint_trajectory[step])
    graph.set_data()
    return graph

with open("yumi_description/yumi.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="")

with open("lift_lego_left.json") as f:
    joint_states = json.load(f)

# Creating the kinematic chain for the left arm
left_arm_chain = kin.create_chain(
    ["yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", 
     "yumi_joint_3_l", "yumi_joint_4_l", "yumi_joint_5_l", 
     "yumi_joint_6_l"],
    "yumi_base_link", "gripper_l_base")  # Assuming 'yumi_tool0_l' is the end effector name for the left arm

filtered_joint_states = filter_joint_states(joint_states, 0.001)
filtered_positions = [state['position'] for state in filtered_joint_states]
reorder = lambda lst: lst[:2] + [lst[-1]] + lst[2:-1]
filtered_left_arm_positions = [reorder(filtered_position[::2]) for filtered_position in filtered_positions]

left_arm_transformations = [left_arm_chain.forward(qpos) for qpos in filtered_left_arm_positions]
trajectory = np.stack(left_arm_transformations)

matrix_list = [
    [ 0.8584588, -0.51253979, -0.01874729, 0.09459746],
    [ 0.51247623, 0.85866017, -0.00841558, -0.41897081],
    [ 0.02041087, -0.00238311, 0.99978884, -0.01390382],
    [ 0., 0., 0., 1.]
]
matrix_np = np.array(matrix_list)

trajectory = matrix_np @ trajectory

# convert to joint space
random_state = np.random.RandomState(0)
joint_trajectory = left_arm_chain.inverse_trajectory(
    trajectory, filtered_left_arm_positions[0], random_state=random_state)

print(np.sum(joint_trajectory-filtered_left_arm_positions))

point_cloud = np.load("PoseEst/pc.npy")
point_cloud_center = np.mean(point_cloud, axis=0)
fig = pv.figure()
fig.plot_vector(point_cloud_center)

T_WC = np.load("handeye/T_WC_head.npy")
intrinsics = np.load("handeye/intrinsics.npy")
fig.plot_camera(M=intrinsics, cam2world=T_WC, sensor_size=(1280, 720))
graph = fig.plot_graph(
    kin.tm, "yumi_base_link", show_visuals=False, show_collision_objects=True,
    show_frames=True, s=0.1, whitelist=["yumi_base_link", "gripper_l_base"])

pv.Trajectory(trajectory, s=0.05).add_artist(fig)


fig.view_init()
fig.animate(
    animation_callback, len(trajectory), loop=True,
    fargs=(graph, left_arm_chain, joint_trajectory))
fig.show()