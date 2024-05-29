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

class ImageListener:
    def __init__(self, dir, video=False):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("d405/color/image_rect_raw", Image, self.image_callback)
        self.last_time = time.time()
        self.fps = 0
        self.demo_rgb = cv2.imread(f"{dir}/demo_wrist_rgb.png")[...,::-1].copy()
        self.demo_rgbs = np.tile(self.demo_rgb, (3, 1, 1, 1))
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 2048)
        self.stable_point = [499.86090324, 103.8931002]

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
        
        live_rgbs = np.empty((3, 480, 848, 3), dtype=np.uint8)
        try:
            for i in range(3):
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
        # err = np.linalg.norm(np.array([x, y]) - np.array(self.stable_point))

        # Calculate FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        # Put texts on the image
        cv2.circle(live_rgb, (int(X/3), int(Y/3)), 5, (0, 0, 255), 3)
        cv2.putText(live_rgb, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(live_rgb, f"ERROR: {err:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.video_writer is not None:
            self.video_writer.write(live_rgb[...,::-1])

        # Display the image using OpenCV
        cv2.imshow("Image Window", live_rgb[...,::-1])
        cv2.waitKey(3)

        # Optionally, save the image to a file
        # cv2.imwrite("/path/to/save/image.jpg", cv_image)

if __name__ == '__main__':
    rospy.init_node('image_listener', anonymous=True)
    il = ImageListener(dir="experiments/scissor")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down image listener node.")
        il.video_writer.release()
    cv2.destroyAllWindows()

