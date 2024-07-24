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
from camera_utils import d405_K as K, d405_T_C_EEF as T_C_EEF, d415_T_WC as T_WC
from trajectory_utils import translation_from_matrix, quaternion_from_matrix, pose_inv, \
                             project3D, create_homogeneous_matrix, apply_transformation_to_waypoints
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lightglue import LightGlue, SuperPoint, match_pair, SIFT
import zipfile
from PIL import Image
from roma import tiny_roma_v1_outdoor
from scipy.spatial.transform import Rotation
from lightglue.utils import load_image, rbd

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

        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 2048)
        self.stable_point = stable_point
        self.key_point_demo = np.array(self.stable_point, dtype=np.float32).reshape(-1, 1, 2)

        # SuperPoint+LightGlue
        # self.extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
        # self.matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

        self.extractor = SIFT(backend='pycolmap', max_num_keypoints=1024).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

        # # Detect keypoints and compute descriptors
        self.feats0 = self.extractor.extract(numpy_image_to_torch(self.demo_rgb))

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

    # def estimate_homography(self, mkpts_0, mkpts_1):
    #     mkpts_0_norm = normalize_points(mkpts_0, K)
    #     mkpts_1_norm = normalize_points(mkpts_1, K)
    
    #     H_norm, _ = cv2.findHomography(mkpts_0_norm, mkpts_1_norm, cv2.USAC_MAGSAC, 0.001, confidence=0.99999, maxIters=1000)
    #     H = K @ H_norm @ np.linalg.inv(K)
    #     key_point_live_hom = cv2.perspectiveTransform(self.key_point_demo, H)
    #     x, y = key_point_live_hom[0][0]

    #     return x, y, H_norm

    def estimate_homography(self, mkpts_0, mkpts_1):
        
        H, _ = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 3.0)
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
        #     mkpts_0, mkpts_1 = self.xfeat.match_xfeat_star(self.demo_rgb, live_rgb, top_k = 2048)
        # except AttributeError as e:
        #     return
        

        try:
            feats1 = self.extractor.extract(numpy_image_to_torch(live_rgb))
            matches01 = self.matcher({'image0': self.feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [self.feats0, feats1, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)
            mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
            mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)
        except AttributeError as e:
            return

        X, Y, H_norm = self.estimate_homography(mkpts_0, mkpts_1)

        # Calculate Error
        err = np.linalg.norm(np.array([X, Y]) - np.array(self.stable_point))

        # Calculate FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time


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
        

if __name__ == '__main__':
    rospy.init_node('image_listener', anonymous=True)
    DIR = "experiments/scissor"

    # Normalize the coordinates to get the 2D image point
    stab_point_2D = [499.86090324, 103.8931002]

    il = ImageListener(dir=DIR, stable_point=stab_point_2D)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down image listener node.")
        il.video_writer.release()

