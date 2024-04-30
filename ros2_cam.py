import rclpy
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile
import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('image_subscriber')

    # Use QoSProfile to get messages that were published while this node was not running
    qos_profile = QoSProfile(depth=10)

    # Define a callback function to handle received images
    def image_callback(msg):
        np_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # Convert BGR to RGB
        np_img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        
        # Save the image
        cv2.imwrite('received_image.jpg', np_img_rgb)

        # Display the received image
        cv2.imshow('Received Image', np_img_rgb)
        cv2.waitKey(1)  # Wait for a short time to process GUI events

    # Subscribe to the image topic
    image_subscription = node.create_subscription(Image, '/D405/color/image_rect_raw', image_callback, qos_profile)

    # Spin the node to receive messages
    rclpy.spin(node)

    # Shutdown ROS 2
    rclpy.shutdown()

if __name__ == '__main__':
    main()


