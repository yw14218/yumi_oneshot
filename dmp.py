import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import json
import time
from movement_primitives.kinematics import Kinematics
from movement_primitives.dmp import CartesianDMP
from trajectory_utils import filter_joint_states

def animation_callback(step, graph, chain, joint_trajectory):
    chain.forward(joint_trajectory[step])
    graph.set_data()
    return graph

with open("yumi_description/yumi.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="")

# Creating the kinematic chain for the left arm
left_arm_chain = kin.create_chain(
    ["yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", 
     "yumi_joint_3_l", "yumi_joint_4_l", "yumi_joint_5_l", 
     "yumi_joint_6_l"],
    "yumi_base_link", "gripper_l_base")  # Assuming 'yumi_tool0_l' is the end effector name for the left arm

with open("lift_lego_left.json") as f:
    joint_states = json.load(f)

filtered_joint_states = filter_joint_states(joint_states, 0.001)
filtered_positions = [state['position'] for state in filtered_joint_states]
reorder = lambda lst: lst[:2] + [lst[-1]] + lst[2:-1]
filtered_left_arm_positions = [reorder(filtered_position[::2]) for filtered_position in filtered_positions]

left_arm_transformations = [left_arm_chain.forward(qpos) for qpos in filtered_left_arm_positions]
trajectoy = np.stack(left_arm_transformations)

# Prepare the time steps (T)
n_steps = len(left_arm_transformations)
T = np.linspace(0, n_steps * 0.1, n_steps)  # Assuming each step is 0.1 seconds apart

# Prepare the state array (Y)
Y = np.zeros((n_steps, 7))
for i, transform in enumerate(left_arm_transformations):
    Y[i, :3] = transform[:3, 3]  # Position
    Y[i, 3:] = pr.quaternion_from_matrix(transform[:3, :3])  # Orientation as quaternion

# Create a Cartesian DMP
dt = 0.01
execution_time = (len(filtered_left_arm_positions) - 1) * dt
dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=10)

# Train the DMP
# Ensure that positions and orientations are formatted as required (e.g., as numpy arrays)
dmp.imitate(T, Y)

# _, Y = dmp.open_loop()
# adapted_trajectory = ptr.transforms_from_pqs(Y)


new_start = Y[0].copy()
new_goal = Y[-1].copy()
# new_goal[1] -= 0.2
# new_goal[2] -= 0.2
dmp.configure(start_y=new_start, goal_y=new_goal)

# Generate a new trajectory towards the new goal
_, Y = dmp.open_loop()

adapted_trajectory = ptr.transforms_from_pqs(Y)


random_state = np.random.RandomState(0)
joint_trajectory = left_arm_chain.inverse_trajectory(
    adapted_trajectory, filtered_left_arm_positions[0], random_state=random_state)

print(np.sum(joint_trajectory-filtered_left_arm_positions))

fig = pv.figure()

graph = fig.plot_graph(
    kin.tm, "yumi_base_link", show_visuals=False, show_collision_objects=True,
    show_frames=True, s=0.1, whitelist=["yumi_base_link", "gripper_l_base"])

pv.Trajectory(adapted_trajectory, s=0.05).add_artist(fig)

fig.view_init()
fig.animate(
    animation_callback, len(adapted_trajectory), loop=True,
    fargs=(graph, left_arm_chain, joint_trajectory))
fig.show()