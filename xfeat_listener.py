#!/usr/bin/env python

import rospy
import cv2
import time
import torch
import numpy as np
import poselib
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge, CvBridgeError
from kornia.geometry.homography import find_homography_dlt
import json
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF, d415_T_WC as T_WC
from trajectory_utils import translation_from_matrix, quaternion_from_matrix, pose_inv, \
                             project3D, create_homogeneous_matrix, apply_transformation_to_waypoints
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lightglue import LightGlue, SuperPoint, match_pair
import zipfile
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from rotation_steerers.steerers import DiscreteSteerer, ContinuousSteerer
from rotation_steerers.matchers.max_similarity import MaxSimilarityMatcher, ContinuousMaxSimilarityMatcher
from PIL import Image
from roma import tiny_roma_v1_outdoor
from gim_lg import GimMatcher
from scipy.spatial.transform import Rotation

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float, device="cuda")

def normalize_points(points, K):
    K_inv = np.linalg.inv(K)
    return (K_inv @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :2]

class ImageListener:
    def __init__(self, dir, stable_point, video=False):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("d405/color/image_rect_raw", ImageMsg, self.image_callback)
        self.last_time = time.time()
        self.fps = 0
        self.demo_rgb = cv2.imread(f"{dir}/demo_wrist_rgb.png")[...,::-1].copy()
        self.demo_seg = cv2.imread(f"{dir}/demo_wrist_seg.png")[...,::-1].astype(bool)
        self.demo_rgb_seg = self.demo_rgb * self.demo_seg
        self.demo_rgb_seg_pil = Image.fromarray(self.demo_rgb_seg)
        self.w_A, self.h_A = self.demo_rgb_seg_pil.size

        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 2048)
        self.stable_point = stable_point
        self.key_point_demo = np.array(self.stable_point, dtype=np.float32).reshape(-1, 1, 2)

        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        # SuperPoint+LightGlue
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

        self.gim_matcher = GimMatcher(self.demo_rgb_seg)

        # self.detector = dedode_detector_L(weights=torch.load("rotation-steerers/model_weights/dedode_detector_L.pth"))
        # self.descriptor = dedode_descriptor_B(weights=torch.load("rotation-steerers/model_weights/B_SO2_Spread_descriptor_setting_B.pth"))
        # self.steerer_order = 8
        # self.steerer = DiscreteSteerer(
        #     generator=torch.matrix_exp(
        #         (2 * 3.14159 / self.steerer_order)
        #         * torch.load("rotation-steerers/model_weights/B_SO2_Spread_steerer_setting_B.pth")
        #     )
        # )
        # self.roma_model = tiny_roma_v1_outdoor(device='cuda')
        # Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
        # with zipfile.ZipFile("EfficientSAM/weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
        #     zip_ref.extractall("weights")
        # self.efficient_sam_vits_model = build_efficient_sam_vits()

        # # Detect keypoints and compute descriptors
        self.kp1, self.des1 = self.sift.detectAndCompute(self.demo_rgb, None)
        # self.kp1, self.des1 = self.orb.detectAndCompute(self.demo_rgb, None)
        self.lowest_error = 1e6
        # self.stable_points = torch.stack(self.stable_point for i in range (5))
        rospy.loginfo("Image listener node initialized.")

        if video:
            # Initialize VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (e.g., 'XVID', 'MJPG')
            frame_size = (848, 480)  # Frame size (width, height)
            fps = 20.0  # Frames per second
            self.video_writer = cv2.VideoWriter('output.avi', fourcc, fps, frame_size)
            if not self.video_writer.isOpened():
                rospy.logerr("Failed to initialize video writer. Check codec and frame size.")
        else:
            self.video_writer = None

        self.x_data, self.y_data = [], []
        self.H_data = []

    def estimate_homography(self, mkpts_0, mkpts_1):
        mkpts_0_norm = normalize_points(mkpts_0, K)
        mkpts_1_norm = normalize_points(mkpts_1, K)
    
        H_norm, _ = cv2.findHomography(mkpts_0_norm, mkpts_1_norm, cv2.USAC_MAGSAC, 0.001, confidence=0.99999, maxIters=1000)
        H = K @ H_norm @ np.linalg.inv(K)
        key_point_live_hom = cv2.perspectiveTransform(self.key_point_demo, H)
        x, y = key_point_live_hom[0][0]

        return x, y, H_norm
    
    def image_callback(self, data):
        
        try:
            live_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        start = time.time()

        # try:
        #     mkpts_0, mkpts_1 = self.gim_matcher.match_images(live_rgb)
        # except AttributeError as e:
        #     return

        # try:
        #     mkpts_0, mkpts_1 = self.xfeat.match_xfeat_star(self.demo_rgb_seg, live_rgb, top_k = 2048)
        # except AttributeError as e:
        #     return

        # try:
        #     warp, certainty = self.roma_model.match(self.demo_rgb_seg_pil, Image.fromarray(live_rgb))
        #     # Sample matches for estimation
        #     matches, certainty = self.roma_model.sample(warp, certainty)
        #     mkpts_0, mkpts_1 = self.roma_model.to_pixel_coordinates(matches, 480, 848, 480, 848)  
        #     mkpts_0 = mkpts_0.cpu().numpy()
        #     mkpts_1 = mkpts_1.cpu().numpy()
        # except AttributeError as e:
        #     return
        
        try:
            feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, numpy_image_to_torch(self.demo_rgb_seg), numpy_image_to_torch(live_rgb))
            matches = matches01['matches']  # indices with shape (K,2)
            mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
            mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)
        except AttributeError as e:
            return

        X, Y, H_norm = self.estimate_homography(mkpts_0, mkpts_1)

        # X, Y, H = self.match_SIFT(live_rgb)
        # X, Y, H = self.match_ORB(live_rgb)
        # X, Y, H = self.match_dedode(live_rgb)

        # print(time.time() - start)

        # Calculate Error
        err = np.linalg.norm(np.array([X, Y]) - np.array(self.stable_point))

        # Calculate FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time

        decompositions = cv2.decomposeHomographyMat(H_norm, np.eye(3))
        best_R, best_t = select_best_decomposition(decompositions)
        drz = best_R[-1]
        # Compute the determinant of H
        det_H = np.linalg.det(H_norm)
        # Compute the ratio Z/Z*
        # Note: Z/Z* = (det(H))^(-1/3)
        Z_ratio = det_H ** (-1/3)
        # Compute the error using natural logarithm
        dz = np.log(Z_ratio)

        # Output the results
        print("Translation (dz):", dz)
        print("Rotation Matrix:\n", np.degrees(drz))

        # Put texts on the image
        cv2.circle(live_rgb, (int(X), int(Y)), 5, (0, 0, 255), 3)
        cv2.putText(live_rgb, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(live_rgb, f"ERROR: {err:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.video_writer is not None:
            self.video_writer.write(live_rgb[...,::-1])

        # Display the image using OpenCV
        cv2.imshow("Image Window", live_rgb[...,::-1])
        cv2.waitKey(3)

        # Optionally, save the image to a file
        # cv2.imwrite("/path/to/save/image.jpg", cv_image)

        self.lowest_error = err if err < self.lowest_error else self.lowest_error
        
        # Update the plot data
        # sum_xy = t_xy[0] + t_xy[1]
        # self.x_data.append(sum_xy)
        # self.y_data.append(err)

        # if len(self.x_data) == 100:
        #     data = {'x_data': self.x_data, 'y_data': self.y_data}
        #     with open('data_lightglue_ori_seg.json', 'w') as outfile:
        #         json.dump(data, outfile)
        #     # Optionally clear the data after saving
        #     self.x_data.clear()
        #     self.y_data.clear()
        self.H_data.append(H_norm)
        if len(self.H_data) == 1200:
            np.save("H_pencile_sharpener_hign_lightglue.npy", np.array(self.H_data))

    def match_SIFT(self, live_img):

        # Detect keypoints and compute descriptors
        kp2, des2 = self.sift.detectAndCompute(live_img, None)

        # Use BFMatcher to find matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des1, des2, k=2)

        # Apply ratio test as per Lowe's paper
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Minimum number of matches to compute homography
        MIN_MATCH_COUNT = 3
        if len(good_matches) > MIN_MATCH_COUNT:
            # Extract location of good matches
            src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Compute homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            raise ValueError(f"Not enough matches found - {len(good_matches)}/{MIN_MATCH_COUNT}")

        # Transform the key point to image 2 using the homography matrix
        key_point_img2_hom = cv2.perspectiveTransform(self.key_point_demo, H)

        # Extract x' and y' coordinates from the result
        x_prime, y_prime = key_point_img2_hom[0][0]

        return x_prime, y_prime, H

    def match_ORB(self, live_img):
        # Detect keypoints and compute descriptors for the live image
        kp2, des2 = self.orb.detectAndCompute(live_img, None)

        # Use BFMatcher to find matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Minimum number of matches to compute homography
        MIN_MATCH_COUNT = 3
        if len(matches) > MIN_MATCH_COUNT:
            # Extract location of good matches
            src_pts = np.float32([self.kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Compute homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Transform the key point to image 2 using the homography matrix
            key_point_img2_hom = cv2.perspectiveTransform(self.key_point_demo, H)

            # Extract x' and y' coordinates from the result
            x_prime, y_prime = key_point_img2_hom[0][0]

            return x_prime, y_prime, H
        else:
            raise ValueError(f"Not enough matches found - {len(matches)}/{MIN_MATCH_COUNT}")
        
    def match_dedode(self, live_img):
        live_img_pil = Image.fromarray(live_img)
        detections_A = self.detector.detect_from_image(self.demo_rgb_seg_pil, num_keypoints = 1024)
        keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]
        detections_B = self.detector.detect_from_image(live_img_pil, num_keypoints = 1024)
        keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]
        matcher = MaxSimilarityMatcher(steerer=self.steerer, steerer_order=self.steerer_order)

        # Describe keypoints and match descriptions (API as in DeDoDe)
        descriptions_A = self.descriptor.describe_keypoints_from_image(self.demo_rgb_seg_pil, keypoints_A)["descriptions"]
        descriptions_B = self.descriptor.describe_keypoints_from_image(live_img_pil, keypoints_B)["descriptions"]

        matches_A, matches_B, batch_ids = matcher.match(
            keypoints_A, descriptions_A,
            keypoints_B, descriptions_B,
            P_A = P_A, P_B = P_B,
            normalize = True, inv_temp=20, threshold = 0.01
        )
        matches_A, matches_B = matcher.to_pixel_coords(
            matches_A, matches_B, 
            self.h_A, self.w_A, self.h_A, self.w_A,
        )

        H, _ = cv2.findHomography(matches_A.cpu().numpy(), matches_B.cpu().numpy(), cv2.USAC_MAGSAC)

        # Transform the key point to image 2 using the homography matrix
        key_point_img2_hom = cv2.perspectiveTransform(self.key_point_demo, H)

        # Extract x' and y' coordinates from the result
        x_prime, y_prime = key_point_img2_hom[0][0]

        return x_prime, y_prime, H

def get_current_stab_3d(T_EEF_World):
    stab_point3d = pose_inv(T_EEF_World @ T_C_EEF) @ T_stab_pose @ T_GRIP_EEF
    # Project the 3D point onto the image plane
    return np.dot(K, stab_point3d[:3, 3])

def select_best_decomposition(decompositions):
    Rs, ts, normals = decompositions[1], decompositions[2], decompositions[3]
    best_index = -1
    best_score = float('inf')

    for i, (R, t, n) in enumerate(zip(Rs, ts, normals)):
        # # Heuristic 1: Check the magnitude of the translation vector (consider all components)
        # translation_magnitude = np.linalg.norm(t)

        # # Heuristic 2: Ensure the normal is close to [0, 0, 1]
        # normal_deviation = np.linalg.norm(n - np.array([0, 0, 1]))

        # Heuristic 3: Check the rotation matrix (for dominant yaw)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        roll = np.arctan2(R[2, 1], R[2, 2])

        # Ideally, pitch and roll should be small in a top-down view
        rotation_score = np.abs(pitch) + np.abs(roll)

        # Combine heuristics into a single score
        score = rotation_score

        if score < best_score:
            best_score = score
            best_index = i

    if best_index != -1:
        best_R = Rs[best_index]
        best_t = ts[best_index]
        best_normal = normals[best_index]
    else:
        print("No valid decomposition found.")

    return Rotation.from_matrix(best_R).as_euler('xyz'), best_t
    
def decompose_homography(H, K):
    # Invert the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Compute the B matrix
    B = np.dot(K_inv, np.dot(H, K))

    # Normalize B so that B[2,2] is 1
    B /= B[2, 2]

    # Extract rotation and translation components using SVD
    U, _, Vt = np.linalg.svd(B)
    R = np.dot(U, Vt)

    if np.linalg.det(R) < 0:
        R = -R

    # Extract translation (last column of B, divided by scale factor)
    t = B[:, 2] / np.linalg.norm(B[:, 0])

    # Normalize translation vector
    t /= np.linalg.norm(t)

    return R, t[:2]  # We only need x, y translation

T_GRIP_EEF = create_homogeneous_matrix([-0.07, 0.08, 0.2], [0, 0, 0, 1])

if __name__ == '__main__':
    rospy.init_node('image_listener', anonymous=True)
    DIR = "experiments/pencile_sharpener"
    file_name = f"{DIR}/demo_bottlenecks.json"

    with open(file_name) as f:
        dbn = json.load(f)
    demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

    bottleneck_left = demo_waypoints[0].tolist()
    
    stab_pose = dbn["bottleneck_left"]
    T_bottleneck_left = create_homogeneous_matrix(bottleneck_left[:3], bottleneck_left[3:])
    T_stab_pose = create_homogeneous_matrix(stab_pose[:3], stab_pose[3:])
    stab_3d_cam = get_current_stab_3d(T_EEF_World=T_bottleneck_left)
    # Normalize the coordinates to get the 2D image point
    stab_point_2D = stab_3d_cam[:2] / stab_3d_cam[2]

    il = ImageListener(dir=DIR, stable_point=stab_point_2D)

    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down image listener node.")
        il.video_writer.release()

