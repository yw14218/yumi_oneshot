import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Function to setup the plot
def setup_plot(ax):
    ax.set_xlabel('∆X')
    ax.set_ylabel('∆Y')
    ax.set_title('Convergence Trajectory on a Sphere (Top View)')
    ax.grid(True)  # Remove grid

# Create a trajectory that converges to the ground truth
def create_convergence_trajectory(initial_pose, ground_truth, steps=50):
    trajectory = np.linspace(initial_pose, ground_truth, steps)
    return trajectory

def create_convergence_trajectory_noise(initial_pose, ground_truth, steps=20):
    # Generate noise for each step
    # Generate trajectory points
    trajectory = np.linspace(initial_pose, ground_truth, steps)
    for i in range(steps):
        noise_x = np.random.normal(0, 0.03, steps)
        noise_y = np.random.normal(0, 0.03, steps)
        noise_yaw = np.random.normal(0, 7, steps)
        
        # Update initial_pose with noisy increments
        trajectory[i][0] += noise_x[i]
        trajectory[i][1] += noise_y[i]
        trajectory[i][2] += noise_yaw[i]
        
    
    return trajectory

# Update function for animation
def update(frame, trajectory, scatter, cmap, norm):
    scatter.set_offsets(trajectory[:frame+1, :2])
    colors = cmap(norm((trajectory[:frame+1, 2])))
    scatter.set_color(colors)
    return scatter,

# Visualization function
def visualize_convergence_on_sphere(trajectory):
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_plot(ax)
    
    # Plot the sphere
    r = 1
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    ax.plot(x, y, color='c', alpha=0.15)

    # Adjust axis limits based on trajectory
    ax.set_xlim(trajectory[:, 0].min(), trajectory[:, 0].max())
    ax.set_ylim(trajectory[:, 1].min(), trajectory[:, 1].max())

    # Initialize scatter plot for trajectory points
    scatter = ax.scatter([], [], color='blue', s=10)

    # Highlight the ground truth
    # ground_truth_coords = ground_truth[0], ground_truth[1]
    # ax.scatter(*ground_truth_coords, color='red', s=100, label='Ground Truth', edgecolor='red', facecolor='none')

    # Create colormap
    cmap = sns.color_palette("hsv", as_cmap=True, n_colors=3600)
    norm = Normalize(vmin=-180, vmax=180)

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(trajectory), fargs=(trajectory, scatter, cmap, norm), interval=100, blit=True)

    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=np.linspace(-180, 180, 7))
    cbar.set_label('∆Yaw (Degrees)')

    plt.legend()
    ani.save('animated_plot.gif', writer='pillow', fps=10)
    plt.show()

# # Ground truth pose (delta_x, delta_y, delta_yaw)
# ground_truth = np.array([0.3, -0.25, 42])

# # Initial prediction (delta_x, delta_y, delta_yaw)
# initial_pose = np.array([0, 0, 0])

# # Generate the convergence trajectory
# trajectory = create_convergence_trajectory_noise(initial_pose, ground_truth)

# # Visualize the convergence on the sphere
# visualize_convergence_on_sphere(trajectory)













