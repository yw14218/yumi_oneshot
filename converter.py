import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

bridge = CvBridge()

def depth_image_callback(msg):
    try:
        # Convert the ROS Image message to an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        print(msg.encoding)
        # Check the image format and convert accordingly
        if cv_image.dtype == np.uint16:
            # Directly publish the image if it is already in the correct format
            converted_msg = bridge.cv2_to_imgmsg(cv_image, encoding="16UC1")
        else:
            rospy.logerr(f"Unsupported image format: {cv_image.dtype}")

        # Publish the converted depth image


    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

if __name__ == "__main__":
    rospy.init_node('depth_image_converter', anonymous=True)
    rospy.Subscriber("/d415/aligned_depth_to_color/image_raw", Image, depth_image_callback)
    pub = rospy.Publisher("/d415/aligned_depth_to_color/image", Image, queue_size=10)
    rospy.spin()
