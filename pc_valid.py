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
