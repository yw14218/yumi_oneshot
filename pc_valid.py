import open3d as o3d
import numpy as np

def icp_registration(source, target, max_correspondence_distance=0.05, max_iteration=100):
    """
    Perform ICP registration using Open3D with robust parameters.
    """
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
        init=np.identity(4)  # Start with identity transform for robustness
    )
    
    return source.transform(icp_result.transformation), icp_result.transformation, icp_result.fitness

def compute_overlap_ratio(source, target, voxel_size=0.01, distance_threshold=0.02):
    """
    Compute the overlap ratio between two point clouds using voxel grid method.
    Adjusted for partial views and potential noise in real-world data.
    """
    # Voxel grid overlap
    source_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(source, voxel_size)
    target_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(target, voxel_size)
    
    source_voxels = set(map(tuple, source_voxel.get_voxels()))
    target_voxels = set(map(tuple, target_voxel.get_voxels()))
    
    voxel_overlap = len(source_voxels.intersection(target_voxels)) / min(len(source_voxels), len(target_voxels))
    
    # Nearest neighbor overlap
    target_tree = o3d.geometry.KDTreeFlann(target)
    nn_overlap_count = 0
    
    for point in source.points:
        [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
        if dist[0] < distance_threshold**2:  # square distance
            nn_overlap_count += 1
    
    nn_overlap_ratio = nn_overlap_count / len(source.points)
    
    return {
        'voxel_overlap_ratio': voxel_overlap,
        'nn_overlap_ratio': nn_overlap_ratio
    }

def centroid_difference(source, target):
    """
    Compute the centroid difference between two point clouds.
    """
    source_centroid = np.mean(np.asarray(source.points), axis=0)
    target_centroid = np.mean(np.asarray(target.points), axis=0)
    return np.linalg.norm(source_centroid - target_centroid)

def detect_off_alignment(overlap_metrics, icp_fitness, centroid_diff, 
                         voxel_threshold=0.3, nn_threshold=0.3, 
                         fitness_threshold=0.6, centroid_threshold=0.1):
    """
    Detect if the alignment is extremely off based on multiple metrics.
    """
    if overlap_metrics['voxel_overlap_ratio'] < voxel_threshold:
        return True, "Voxel overlap ratio is too low"
    if overlap_metrics['nn_overlap_ratio'] < nn_threshold:
        return True, "Nearest neighbor overlap ratio is too low"
    if icp_fitness < fitness_threshold:
        return True, "ICP fitness score is too low"
    if centroid_diff > centroid_threshold:
        return True, "Centroid difference is too high"
    return False, "Alignment seems reasonable"

def robust_alignment_check(source, target):
    """
    Perform a robust alignment check with fallback to centroid difference.
    """
    try:
        # Attempt ICP registration
        transformed_source, _, icp_fitness = icp_registration(source, target)
        
        # Compute overlap metrics
        overlap_metrics = compute_overlap_ratio(transformed_source, target)
        
        # Compute centroid difference as a fallback
        centroid_diff = centroid_difference(transformed_source, target)
        
        # Detect if alignment is extremely off
        is_off, reason = detect_off_alignment(overlap_metrics, icp_fitness, centroid_diff)
        
        return is_off, reason, {
            'icp_fitness': icp_fitness,
            'centroid_difference': centroid_diff,
            **overlap_metrics
        }
    except Exception as e:
        print(f"Error in alignment check: {str(e)}")
        # Fallback to centroid difference
        centroid_diff = centroid_difference(source, target)
        is_off = centroid_diff > 0.1  # Adjust this threshold as needed
        return is_off, "Fallback to centroid difference", {'centroid_difference': centroid_diff}
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
from scipy.spatial.transform import Rotation as R

class PBVS_UKF:
    def __init__(self, initial_state, d405_T_C_EEF, cap_t, cap_r):
        self.cap_t = cap_t
        self.cap_r = cap_r
        self.d405_T_C_EEF = d405_T_C_EEF  # Hand-eye calibration matrix
        
        # Define UKF parameters
        dim_x = 6  # State dimension [x, y, z, roll, pitch, yaw]
        dim_z = 6  # Measurement dimension
        points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=-3)
        self.ukf = UKF(dim_x=dim_x, dim_z=dim_z, fx=self.fx, hx=self.hx, points=points)
        self.ukf.x = initial_state  # Initial state (relative pose)
        self.ukf.P *= 0.1  # Initial covariance
        self.ukf.Q *= 0.01  # Process noise
        self.base_R = np.eye(6) * 0.01  # Base measurement noise
        
        self.distance_threshold = 0.5  # Threshold for adaptive measurement noise
        self.R_scale_factor = 10  # Maximum scale factor for measurement noise
        self.gain = 0.1  # Control gain

    def fx(self, x, dt):
        """State transition function: simple constant pose model."""
        return x  # Assuming no inherent dynamics, control input applied separately

    def hx(self, x):
        """Measurement function: convert relative end-effector pose to camera frame."""
        T_eef = self.state_to_transform(x)
        T_cam = np.linalg.inv(self.d405_T_C_EEF) @ T_eef @ self.d405_T_C_EEF
        return self.transform_to_state(T_cam)

    def state_to_transform(self, state):
        """Convert state vector [x, y, z, roll, pitch, yaw] to transformation matrix."""
        t = state[:3]
        r = R.from_euler('xyz', state[3:], degrees=False).as_matrix()
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t
        return T

    def transform_to_state(self, T):
        """Convert transformation matrix to state vector [x, y, z, roll, pitch, yaw]."""
        t = T[:3, 3]
        r = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
        return np.concatenate((t, r))

    def update_measurement_noise(self, distance):
        """Update measurement noise covariance based on distance to workspace."""
        scale = 1 + (self.R_scale_factor - 1) * min(max(distance - self.distance_threshold, 0) / self.distance_threshold, 1)
        self.ukf.R = self.base_R * scale

    def compute_control_input(self, state):
        """Compute control input based on the current state estimate."""
        translation = state[:3]
        rotation = state[3:]
        control_input = np.concatenate([
            np.clip(self.gain * translation, -self.cap_t, self.cap_t),
            np.clip(self.gain * rotation, -self.cap_r, self.cap_r)
        ])
        return control_input

    def run(self):
        error = float('inf')
        trajectory = []
        num_iteration = 0

        while error > 0.005:
            # Step 1: Predict the next state
            self.ukf.predict()
            
            # Step 2: Get new measurement
            T_delta_cam = self.estimate_rel_pose()  # This function needs to be implemented
            measurement = self.transform_to_state(T_delta_cam)
            
            # Step 3: Update measurement noise based on distance
            distance = np.linalg.norm(measurement[:3])
            self.update_measurement_noise(distance)
            
            # Step 4: Update the filter with the new measurement
            self.ukf.update(measurement)
            
            # Step 5: Get the current state estimate
            current_state = self.ukf.x
            
            # Step 6: Compute control input based on the current state estimate
            control_input = self.compute_control_input(current_state)
            
            # Step 7: Apply control input to the robot
            T_eef_world = yumi.get_curent_T_left()
            T_delta_eef = self.state_to_transform(control_input)
            T_eef_world_new = T_eef_world @ T_delta_eef
            
            # Extract new pose
            new_position = T_eef_world_new[:3, 3]
            new_orientation = R.from_matrix(T_eef_world_new[:3, :3]).as_euler('xyz', degrees=False)
            
            # Construct the new end-effector pose
            eef_pose = yumi.create_pose_euler(new_position[0], new_position[1], new_position[2],
                                              new_orientation[0], new_orientation[1], new_orientation[2])
            
            # Move the end-effector to the new pose
            self.cartesian_controller.move_eef(eef_pose)
            
            # Record the trajectory for analysis
            trajectory.append([new_position[0], new_position[1], np.degrees(new_orientation[2])])
            
            # Update error and iteration count
            error = np.linalg.norm(current_state[:3])  # Use the estimated relative position as error
            num_iteration += 1
            print(f"Iteration {num_iteration}: Error {error}")

        return trajectory
        
# Example usage
if __name__ == "__main__":
    # Load or create sample point clouds
    # Replace this with your actual point cloud loading code
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    
    # Fill with some sample data (replace this with your actual point cloud data)
    # Simulating partial, noisy data from an overhead camera
    source.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3) * 0.5)
    target.points = o3d.utility.Vector3dVector(np.random.rand(800, 3) * 0.5 + np.array([0.1, 0.1, 0]))
    
    # Add some noise to simulate real-world conditions
    source.points = o3d.utility.Vector3dVector(np.asarray(source.points) + np.random.randn(1000, 3) * 0.01)
    target.points = o3d.utility.Vector3dVector(np.asarray(target.points) + np.random.randn(800, 3) * 0.01)
    
    # Perform robust alignment check
    is_off, reason, metrics = robust_alignment_check(source, target)
    
    print("Alignment Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\nIs alignment extremely off? {'Yes' if is_off else 'No'}")
    print(f"Reason: {reason}")
