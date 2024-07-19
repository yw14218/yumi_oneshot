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

def find_transformation(X, Y):
    """
    Find the optimal rigid transformation (rotation and translation) between two sets of 3D points.

    This function calculates the rotation matrix and translation vector that align two sets of 3D points
    by minimizing the mean squared error between the transformed points in set X and the corresponding points in set Y.

    The transformation is computed using the following steps:
    1. Remove zero vectors from both point sets.
    2. Compute the centroids of the remaining points in each set.
    3. Center the points by subtracting their respective centroids.
    4. Compute the covariance matrix between the centered point sets.
    5. Perform Singular Value Decomposition (SVD) on the covariance matrix to obtain the rotation matrix.
    6. Compute the translation vector.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (N, 3) representing the first set of 3D points.
    Y (numpy.ndarray): A 2D array of shape (N, 3) representing the second set of 3D points.

    Returns:
    R (numpy.ndarray): A 2D array of shape (3, 3) representing the rotation matrix.
    t (numpy.ndarray): A 1D array of shape (3,) representing the translation vector.

    Raises:
    ValueError: If the input arrays X and Y do not have the same shape or do not have three columns.
    """
    if X.shape != Y.shape or X.shape[1] != 3:
        raise ValueError("Input arrays X and Y must have the same shape and must be 2D arrays with 3 columns.")

    # Calculate centroids
    non_zero_indices = np.all(X != [0, 0, 0], axis=1) & np.all(Y != [0, 0, 0], axis=1)
    X = X[non_zero_indices]
    Y = Y[non_zero_indices]
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Determine translation vector
    t = cY - np.dot(R, cX)

    return R, t

def solve_transform_3d(mkpts_0, mkpts_1, depth_ref, depth_cur):
    """
    Compute the 3D transformation matrix between two sets of 3D points derived from depth maps.

    This function estimates the transformation between two sets of 3D points (from depth maps) by:
    1. Converting 2D keypoints and depth maps into 3D points.
    2. Calculating the rigid transformation (rotation and translation) between the two sets of 3D points.
    3. Constructing a 4x4 transformation matrix from the computed rotation and translation.

    Parameters:
    mkpts_0 (numpy.ndarray): A 2D array of shape (N, 2) containing 2D keypoints from the reference image.
    mkpts_1 (numpy.ndarray): A 2D array of shape (N, 2) containing 2D keypoints from the current image.
    depth_ref (numpy.ndarray): A 2D array representing the depth map of the reference image.
    depth_cur (numpy.ndarray): A 2D array representing the depth map of the current image.

    Returns:
    T_est (numpy.ndarray): A 4x4 transformation matrix that includes the rotation and translation between the two sets of 3D points.

    Raises:
    ValueError: If the shapes of `mkpts_0` and `mkpts_1` do not match or if the depth maps and keypoints are incompatible.
    """
    # Convert 2D keypoints and depth maps to 3D points
    points1_3d = add_depth(mkpts_0, depth_ref)
    points2_3d = add_depth(mkpts_1, depth_cur)
    
    # Compute the transformation between the two sets of 3D points
    delta_R_camera, delta_t_camera = find_transformation(points1_3d, points2_3d)
    
    # Create a 4x4 transformation matrix
    T_est = np.eye(4)
    T_est[:3, :3] = delta_R_camera
    T_est[:3, 3] = delta_t_camera
    
    return T_est
