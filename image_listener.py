#!/usr/bin/env python

import rospy
import cv2
import time
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageListener:
    def __init__(self, dir, video=False):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("d405/color/image_rect_raw", Image, self.image_callback)
        self.last_time = time.time()
        self.fps = 0
        self.demo_rgb = cv2.imread(f"{dir}/demo_wrist_rgb.png")[...,::-1].copy()
        self.stable_point = [499.86090324, 103.8931002]
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

    def image_callback(self, data):

        try:
            live_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Calculate FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        # Put texts on the image
        cv2.circle(live_rgb, (int(self.stable_point[0]), int(self.stable_point[1])), 5, (0, 0, 255), 3)
        cv2.putText(live_rgb, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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