import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state

d405_rgb_topic_name = "d405/color/image_rect_raw"
d405_depth_topic_name = "d405/aligned_depth_to_color/image_raw"
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
    
    return points_3ds, normalized_mkpts

def compute_transform_least_square(X, Y):
    """
    Find the optimal rigid transformation (rotation and translation) between two sets of 3D points with least squares.

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

def find_transformation(X, Y, max_iterations=100, threshold=0.005, random_state=None, adaptive=False):
    """
    Find the optimal rigid transformation (rotation and translation) between two sets of 3D points using RANSAC.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (N, 3) representing the first set of 3D points.
    Y (numpy.ndarray): A 2D array of shape (N, 3) representing the second set of 3D points.
    max_iterations (int): Maximum number of RANSAC iterations.
    threshold (float): Inlier threshold for RANSAC.
    random_state (int or None): Seed for random number generator.
    adaptive (bool): Whether to use an adaptive threshold if no transformation is found.

    Returns:
    R (numpy.ndarray or None): A 2D array of shape (3, 3) representing the rotation matrix, or None if not enough points.
    t (numpy.ndarray or None): A 1D array of shape (3,) representing the translation vector, or None if not enough points.
    inliers (numpy.ndarray or None): A 1D boolean array indicating which points are inliers, or None if not enough points.
    """
    if X.shape != Y.shape or X.shape[1] != 3:
        raise ValueError("Input arrays X and Y must have the same shape and must be 2D arrays with 3 columns.")
    
    num_points = X.shape[0]
    
    if num_points < 3:
        # Not enough points to define a transformation
        print("Not enough points to compute a unique transformation.")
        return None, None, None

    rng = check_random_state(random_state)
    best_inliers = None
    best_R = None
    best_t = None
    best_error = float('inf')

    min_points = 4 if num_points >= 4 else num_points

    for i in range(max_iterations):
        # Randomly sample min_points points (or fewer if less than 4 points available)
        indices = rng.choice(num_points, size=min_points, replace=False)
        X_sample = X[indices]
        Y_sample = Y[indices]

        # Skip iteration if we somehow end up with empty samples
        if X_sample.shape[0] == 0 or Y_sample.shape[0] == 0:
            continue

        # Calculate transformation for the sample
        R, t = compute_transform_least_square(X_sample, Y_sample)

        if R is None or t is None:
            continue  # Skip if no valid transformation was computed

        # Transform all points using the estimated rotation and translation
        Y_pred = np.dot(X, R.T) + t

        # Compute inliers based on threshold
        errors = np.linalg.norm(Y_pred - Y, axis=1)
        inliers = errors < threshold

        # Skip if no inliers found
        if np.sum(inliers) == 0:
            continue

        # Check if current model is better
        current_error = mean_squared_error(Y[inliers], Y_pred[inliers])
        if best_inliers is None or np.sum(inliers) > np.sum(best_inliers) or (np.sum(inliers) == np.sum(best_inliers) and current_error < best_error):
            best_inliers = inliers
            best_R = R
            best_t = t
            best_error = current_error

    if best_R is None or best_t is None:
        print("RANSAC failed to find a valid transformation with the initial threshold.")

        # Fallback mechanism: Adaptive threshold or using all points
        if adaptive:
            print("Attempting with an increased threshold.")
            threshold *= 10  # Increase the threshold and retry
            return find_transformation(X, Y, max_iterations, threshold, random_state, adaptive=False)
        else:
            print("Fallback: Computing transformation using all points.")
            R, t = compute_transform_least_square(X, Y)
            return R, t

    return best_R, best_t

def solve_transform_3d(mkpts_0, mkpts_1, depth_ref, depth_cur, K, compute_homography=True):
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
    mkpts_0_3d, normalize_mkpts_0 = add_depth(mkpts_0, depth_ref, K)
    mkpts_1_3d, normalize_mkpts_1 = add_depth(mkpts_1, depth_cur, K)
    
    # Compute the transformation between the two sets of 3D points
    delta_R_camera, delta_t_camera = find_transformation(mkpts_0_3d, mkpts_1_3d)

    H_norm = None
    if compute_homography is True:
        H_norm, _ = cv2.findHomography(normalize_mkpts_0, normalize_mkpts_1, cv2.USAC_MAGSAC, 0.001, confidence=0.99999)
        if H_norm is not None:
            H = K @ H_norm @ np.linalg.inv(K)
            selected_R, selected_t = homography_test(H, mkpts_0, mkpts_1, K)
            # If a valid homography decomposition exists, use it and scale the translation
            if selected_R is not None and selected_t is not None:
                delta_R_camera = selected_R
                # delta_t_camera = selected_t * (np.linalg.norm(delta_t_camera) / np.linalg.norm(selected_t))

        # Create a 4x4 transformation matrix
    T_est = np.eye(4)
    T_est[:3, :3] = delta_R_camera
    T_est[:3, 3] = delta_t_camera
    
    return T_est, H_norm

def homography_test(H, mkpts_0, mkpts_1, K, inlier_ratio_threshold = 0.5):
    """
    Perform a homography test based on reprojection error and decompose
    the homography matrix into possible rotations and translations.

    Args:
        H (np.ndarray): 3x3 homography matrix.
        mkpts_0 (np.ndarray): Matched keypoints from image 0 (Nx2).
        mkpts_1 (np.ndarray): Matched keypoints from image 1 (Nx2).
        K (np.ndarray): Camera intrinsic matrix.

    Returns:
        T_delta_cam (np.ndarray or None): 4x4 transformation matrix if the homography is valid,
                                          otherwise None.
    """
    
    # Validate the homography using inlier ratio
    inlier_ratio = homography_test_inlier_ratio(H, mkpts_0, mkpts_1)
    
    # Set a threshold for the inlier ratio
    inlier_ratio_threshold = 0.3

    if inlier_ratio >= inlier_ratio_threshold:
        print(f"Valid Homography with inlier_ratio: {inlier_ratio:.6f}")

         # Decompose the homography matrix into possible rotation matrices, translation vectors, and normals
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

        best_solution_index = None
        best_solution_score = float('inf')

        # Iterate over the decompositions and select the best one based on specific criteria
        for i in range(num_solutions):
            R = rotations[i]
            t = translations[i]
            
            # Check if the Z-component of the translation vector is positive (positive depth)
            if t[2] <= 0:
                continue

            # Calculate yaw (rotation around the Z-axis)
            yaw = np.arctan2(R[1, 0], R[0, 0])

            # Calculate the score (penalize less yaw or negative depth)
            score = abs(yaw)

            # Select the best solution based on score
            if score < best_solution_score:
                best_solution_score = score
                best_solution_index = i

        if best_solution_index is not None:
            selected_R = rotations[best_solution_index]
            selected_t = translations[best_solution_index]

            return selected_R, selected_t.flatten()
        else:
            print("No valid solution found after homography decomposition.")

    else:
        print(f"Homography hypothesis rejected due to low inlier ratio: {inlier_ratio:.3f}")

    # Fallback if homography is not valid or no valid decomposition was found
    return None, None

def homography_test_inlier_ratio(H, mkpts_0, mkpts_1, threshold=5.0):
    """
    Validate homography using inlier ratio instead of reprojection error.

    Args:
        H (np.ndarray): 3x3 homography matrix.
        mkpts_0 (np.ndarray): Matched keypoints from image 0 (Nx2).
        mkpts_1 (np.ndarray): Matched keypoints from image 1 (Nx2).
        threshold (float): Distance threshold to consider a point as an inlier.

    Returns:
        (float): Inlier ratio, fraction of points that are consistent with the homography.
    """
    # Reproject points from mkpts_0 to mkpts_1 using homography H
    reprojected_points = cv2.perspectiveTransform(mkpts_0.reshape(-1, 1, 2), H).reshape(-1, 2)
    
    # Compute Euclidean distances between the original and reprojected points
    distances = np.linalg.norm(reprojected_points - mkpts_1, axis=1)
    
    # Determine inliers based on the threshold
    inliers = distances < threshold
    inlier_count = np.sum(inliers)
    total_count = len(mkpts_0)
    
    # Compute inlier ratio
    inlier_ratio = inlier_count / total_count
    
    return inlier_ratio