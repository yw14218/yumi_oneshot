import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from movement_primitives.kinematics import Kinematics
from scipy.spatial.transform import Rotation as R

def animation_callback(step, graph, chain, joint_trajectory):
    chain.forward(joint_trajectory[step])
    graph.set_data()
    return graph

def generate_hemisphere_poses_around_point(object_coords, radius, num_viewpoints):

    inclinations = np.linspace(0, np.pi, num_viewpoints)
    azimuths = np.linspace(0, 2*np.pi, num_viewpoints)
    
    # Generate camera poses around the hemisphere
    camera_poses = []
    for theta in inclinations:
        for phi in azimuths:
            x = radius * np.sin(theta) * np.cos(phi) + object_coords[0]
            y = radius * np.sin(theta) * np.sin(phi) + object_coords[1]
            z = radius * np.cos(theta) + object_coords[2]
            
            # Calculate the direction vector from the camera to the object
            direction = object_coords - np.array([x, y, z])
            
            # Normalize the direction vector
            direction = direction / np.linalg.norm(direction)
            
            # Convert the direction vector to Euler angles
            pitch = np.arcsin(-direction[2])
            roll = np.arccos(np.clip(direction[0] / np.cos(pitch), -1, 1))
            yaw = np.arccos(np.clip(direction[1] / np.cos(pitch), -1, 1))
            
            q = pr.quaternion_from_axis_angle([roll, pitch, yaw])
            camera_poses.append([x, y, z, q[0], q[1], q[2], q[3]])

    return camera_poses

with open("yumi_description/yumi.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="")

# Creating the kinematic chain for the left arm
left_arm_chain = kin.create_chain(
    ["yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", 
     "yumi_joint_3_l", "yumi_joint_4_l", "yumi_joint_5_l", 
     "yumi_joint_6_l"],
    "yumi_base_link", "gripper_l_base")  # Assuming 'yumi_tool0_l' is the end effector name for the left arm

fig = pv.figure()

graph = fig.plot_graph(
    kin.tm, "yumi_base_link", show_visuals=False, show_collision_objects=True,
    show_frames=True, s=0.1, whitelist=["yumi_base_link", "gripper_l_base"])


fig.view_init()
x = ptr.transforms_from_pqs([0.5, 0, 0.15, 0, 0, 0, 1])
fig.plot_transform(x, s=0.1)
camera_poses = generate_hemisphere_poses_around_point([0.5, 0, 0.15], 0.2, 10)
for camera_pose in camera_poses:
    x = ptr.transforms_from_pqs(camera_pose)
    fig.plot_transform(x, s=0.1)


fig.show()