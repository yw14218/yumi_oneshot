import numpy as np

d405_K = np.load("handeye/intrinsics_d405.npy")
d415_K = np.load("handeye/intrinsics_d415.npy")
d405_T_C_EEF = np.load("handeye/T_C_EEF_wrist_l.npy")
d415_T_WC = np.load("handeye/T_WC_head.npy")

def normalize_mkpts(mkpts, K):
    """
    Normalize image points using the camera intrinsic matrix.

    Args:
    mkpts: Nx2 numpy array of image points.
    K: 3x3 intrinsic matrix.

    Returns:
    normalized_points: Nx3 numpy array of normalized points in homogeneous coordinates.
    """
    # Convert points to homogeneous coordinates
    mkpts_h = np.hstack((mkpts, np.ones((mkpts.shape[0], 1))))

    # Normalize points
    normalized_points = (np.linalg.inv(K) @ mkpts_h.T).T

    return normalized_points

def add_depth(mkpts, depth, K):
    """
    Converts 2D points to 3D points using depth information.

    Parameters:
    mkpts (np.ndarray): Array of 2D points with shape (N, 2), where N is the number of points.
    depth (np.ndarray): Depth map with shape (H, W), where H and W are the height and width of the depth map.
    K (np.ndarray): Camera intrinsic matrix with shape (3, 3).

    Returns:
    np.ndarray: Array of 3D points with shape (N, 3).
    """
    normalized_mkpts = normalize_mkpts(mkpts, K)
    depth_values = depth[mkpts[:, 1].astype(int), mkpts[:, 0].astype(int)]
    points_3ds = normalized_mkpts * depth_values[:, np.newaxis] / 1000
    
    return points_3ds