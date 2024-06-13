#!/usr/bin/env python

import rospy
import cv2
import time
import torch
import numpy as np
import poselib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from kornia.geometry.homography import find_homography_dlt
import json
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF, d415_T_WC as T_WC
from trajectory_utils import translation_from_matrix, quaternion_from_matrix, pose_inv, \
                             project3D, create_homogeneous_matrix, apply_transformation_to_waypoints
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lightglue import LightGlue, SuperPoint, match_pair
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vits
import zipfile

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float, device="cuda")

class ImageListener:
    def __init__(self, dir, stable_point, video=False):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("d405/color/image_rect_raw", Image, self.image_callback)
        self.last_time = time.time()
        self.fps = 0
        self.demo_rgb = cv2.imread(f"{dir}/demo_wrist_rgb.png")[...,::-1].copy()
        self.demo_seg = cv2.imread(f"{dir}/demo_wrist_seg.png")[...,::-1].astype(bool)
        self.demo_rgb_seg = self.demo_rgb * self.demo_seg

        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 2048)
        self.stable_point = stable_point
        self.key_point_demo = np.array(self.stable_point, dtype=np.float32).reshape(-1, 1, 2)
        self.sift = cv2.SIFT_create()

        # SuperPoint+LightGlue
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

        # Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
        with zipfile.ZipFile("EfficientSAM/weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
            zip_ref.extractall("weights")
        self.efficient_sam_vits_model = build_efficient_sam_vits()

        # Detect keypoints and compute descriptors
        self.kp1, self.des1 = self.sift.detectAndCompute(self.demo_rgb, None)

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

    def estimate_homography(self, mkpts_0, mkpts_1):
        H, _ = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC)
        key_point_live_hom = cv2.perspectiveTransform(self.key_point_demo, H)
        x, y = key_point_live_hom[0][0]

        return x, y, H
    
    def image_callback(self, data):
        
        try:
            live_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        start = time.time()

        # try:
        #     mkpts_0, mkpts_1 = self.xfeat.match_xfeat_star(self.demo_rgb_seg, self.demo_rgb_seg, top_k = 2048)
        # except AttributeError as e:
        #     return

        try:
            feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, numpy_image_to_torch(self.demo_rgb_seg), numpy_image_to_torch(live_rgb))
            matches = matches01['matches']  # indices with shape (K,2)
            mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
            mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)
        except AttributeError as e:
            return

        X, Y, H = self.estimate_homography(mkpts_0, mkpts_1)
        # X, Y, H = self.match_SIFT(live_rgb)

        # print(numpy_image_to_torch(live_rgb)[None, ...].device)
        # print(feats1['keypoints'][matches[..., 1]].to(device='cuda').device)
        # print(torch.tensor([[[1, 1]]]).to(device='cuda').device)
        # predicted_logits, predicted_iou = self.efficient_sam_vits_model(
        #     numpy_image_to_torch(live_rgb)[None, ...].to(device='cpu'),
        #     feats1['keypoints'][matches[..., 1]].to(device='cpu'),
        #     torch.tensor([[[1, 1]]]).to(device='cpu'),
        # )
        # sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        # predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        # predicted_logits = torch.take_along_dim(
        #     predicted_logits, sorted_ids[..., None, None], dim=2
        # )
        # # The masks are already sorted by their predicted IOUs.
        # # The first dimension is the batch size (we have a single image. so it is 1).
        # # The second dimension is the number of masks we want to generate (in this case, it is only 1)
        # # The third dimension is the number of candidate masks output by the model.
        # # For this demo we use the first mask.
        # mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
        # masked_image_np = live_rgb.copy().astype(np.uint8) * mask[:,:,None]

        # print(time.time() - start)

        # Calculate Error
        err = np.linalg.norm(np.array([X, Y]) - np.array(self.stable_point))

        # Calculate FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        R, t_xy = decompose_homography(H, K)
        # Output the results
        print("Translation (x, y):", t_xy)
        print("Rotation Matrix:\n", np.degrees(np.arctan2(H[1, 0], H[0, 0])))

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
        sum_xy = t_xy[0] + t_xy[1]
        self.x_data.append(sum_xy)
        self.y_data.append(err)

        if len(self.x_data) == 100:
            data = {'x_data': self.x_data, 'y_data': self.y_data}
            with open('data_lightglue_ori_seg.json', 'w') as outfile:
                json.dump(data, outfile)
            # Optionally clear the data after saving
            self.x_data.clear()
            self.y_data.clear()


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


def get_current_stab_3d(T_EEF_World):
    stab_point3d = pose_inv(T_EEF_World @ T_C_EEF) @ T_stab_pose @ T_GRIP_EEF
    # Project the 3D point onto the image plane
    return np.dot(K, stab_point3d[:3, 3])

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

T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.136], [0, 0, 0, 1])

if __name__ == '__main__':
    rospy.init_node('image_listener', anonymous=True)
    DIR = "experiments/pencile_sharpener"
    file_name = f"{DIR}/demo_bottlenecks.json"

    with open(file_name) as f:
        dbn = json.load(f)
    demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

    bottleneck_left = demo_waypoints[0].tolist()
    bottleneck_right = demo_waypoints[1].tolist()
    stab_pose = dbn["grasp_right"]
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
    cv2.destroyAllWindows()

