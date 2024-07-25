import abc
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Lock, Condition
from camera_utils import d405_rgb_topic_name, d405_depth_topic_name
from moveit_utils.cartesian_control import YuMiLeftArmCartesianController
from lightglue import LightGlue, SIFT
from lightglue.utils import numpy_image_to_torch, rbd

class CartesianVisualServoer(abc.ABC):
    def __init__(self, use_depth=False):        
        self.bridge = CvBridge()
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.cartesian_controller = YuMiLeftArmCartesianController()
        self.rgb_subscriber = rospy.Subscriber(d405_rgb_topic_name, Image, self.image_callback)

        self.images = {
            "rgb": None,
            "depth": None
        }
        self.use_depth = use_depth

        if self.use_depth:
            self.depth_subscriber = rospy.Subscriber(d405_depth_topic_name, Image, self.image_callback)

    def image_callback(self, msg):
        with self.lock:
            if msg.header.frame_id == d405_rgb_topic_name:
                self.images["rgb"] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            elif self.use_depth and msg.header.frame_id == d405_depth_topic_name:
                self.images["depth"] = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.condition.notify_all()

    def observe(self):
        with self.lock:
            self.images["rgb"] = None
            if self.use_depth:
                self.images["depth"] = None

            collected = self.condition.wait_for(
                lambda: self.images["rgb"] is not None and (not self.use_depth or self.images["depth"] is not None),
                timeout=1
            )
            
            if not collected:
                rospy.logwarn("Timeout occurred while waiting for images.")
                return (None, None) if self.use_depth else None

            live_rgb = self.images["rgb"]
            live_depth = self.images["depth"] if self.use_depth else None
            self.images["rgb"] = None
            if self.use_depth:
                self.images["depth"] = None

        return (live_rgb, live_depth) if self.use_depth else (live_rgb, None)
    
    @abc.abstractmethod
    def run(self):
        """Run the visual servoing loop."""
        pass

class SiftLightGlueVisualServoer(CartesianVisualServoer):
    def __init__(self, rgb_ref, seg_ref, use_depth=False):
        super().__init__(use_depth=use_depth)

        self.extractor = SIFT(backend='pycolmap', max_num_keypoints=1024).eval().cuda()
        self.matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()
        self.feats0 = self.extractor.extract(numpy_image_to_torch(rgb_ref))
        self.seg_ref = seg_ref

    def match_siftlg(self, filter_seg=True):
        live_rgb, live_depth = self.observe()

        if live_rgb is None:
            raise RuntimeError("No image received. Please check the camera and topics.")

        feats1 = self.extractor.extract(numpy_image_to_torch(live_rgb))
        matches01 = self.matcher({'image0': self.feats0, 'image1': feats1})

        feats0, feats1, matches01 = [self.rbd(x) for x in [self.feats0, feats1, matches01]]
        matches = matches01['matches']

        mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
        mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

        if filter_seg:
            coords = mkpts_0.astype(int)
            seg_values = self.seg_ref[coords[:, 1], coords[:, 0]]
            mask = seg_values

            mkpts_0 = mkpts_0[mask]
            mkpts_1 = mkpts_1[mask]

        return mkpts_0, mkpts_1, live_depth
    
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative