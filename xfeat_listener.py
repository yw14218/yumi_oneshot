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

class ImageListener:
    def __init__(self, dir, stable_point, video=False):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("d405/color/image_rect_raw", Image, self.image_callback)
        self.last_time = time.time()
        self.fps = 0
        self.demo_rgb = cv2.imread(f"{dir}/demo_wrist_rgb.png")[...,::-1].copy()
        self.demo_rgbs = np.tile(self.demo_rgb, (2, 1, 1, 1))
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 2048)
        self.stable_point = stable_point

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

    def estimate_homography(self, mkpts):
        mkpts_0_np = mkpts[:, :2].cpu().numpy().reshape(-1, 2)  # Convert tensor to numpy array
        mkpts_1_np = mkpts[:, 2:].cpu().numpy().reshape(-1, 2)  # Convert tensor to numpy array
        H, _ = poselib.estimate_homography(mkpts_0_np, mkpts_1_np)
        key_point_demo = np.array(self.stable_point, dtype=np.float32).reshape(-1, 1, 2)
        key_point_live_hom = cv2.perspectiveTransform(key_point_demo, H)
        x, y = key_point_live_hom[0][0]

        return x, y
    
    def image_callback(self, data):
        
        live_rgbs = np.empty((2, 480, 848, 3), dtype=np.uint8)
        try:
            for i in range(2):
                live_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
                live_rgbs[i] = live_rgb
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        start = time.time()
        try:
            mkpts_list = self.xfeat.match_xfeat_star(self.demo_rgbs, live_rgbs, top_k = 2048)
        except AttributeError as e:
            return

        X = 0
        Y = 0
        for mkpts in mkpts_list:
            x, y = self.estimate_homography(mkpts)
            X += x
            Y += y
        # tensor_batch = torch.stack(mkpts_list)
        # Hs = find_homography_dlt(tensor_batch[:, :, :2], tensor_batch[:, :, 2:])
 
        # for H in Hs:
        #     H = H.cpu().numpy()
        #     key_point_demo = np.array(self.stable_point, dtype=np.float32).reshape(-1, 1, 2)
        #     key_point_live_hom = cv2.perspectiveTransform(key_point_demo, H)
        #     x, y = key_point_live_hom[0][0]
        #     xs += x
        #     ys += y

        print(time.time() - start)
        # Calculate Error
        err = np.linalg.norm(np.array([X/2, Y/2]) - np.array(self.stable_point))

        # Calculate FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        # Put texts on the image
        cv2.circle(live_rgb, (int(X/2), int(Y/2)), 5, (0, 0, 255), 3)
        cv2.putText(live_rgb, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(live_rgb, f"ERROR: {err:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.video_writer is not None:
            self.video_writer.write(live_rgb[...,::-1])

        # Display the image using OpenCV
        cv2.imshow("Image Window", live_rgb[...,::-1])
        cv2.waitKey(3)

        # Optionally, save the image to a file
        # cv2.imwrite("/path/to/save/image.jpg", cv_image)

def get_current_stab_3d(T_EEF_World):
    stab_point3d = pose_inv(T_EEF_World @ T_C_EEF) @ T_stab_pose @ T_GRIP_EEF
    # Project the 3D point onto the image plane
    return np.dot(K, stab_point3d[:3, 3])

T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.100], [0, 0, 0, 1])

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

