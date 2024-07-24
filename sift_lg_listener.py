import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Lock, Condition
from lightglue import LightGlue, SIFT
from lightglue.utils import numpy_image_to_torch, rbd

class SiftLightGlueListener():
    def __init__(self, rgb_ref, seg_ref):
        self.live_rgb_image = None
        self.bridge = CvBridge()
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.subscriber = rospy.Subscriber("d405/color/image_rect_raw", Image, self.image_callback)
        self.extractor = SIFT(backend='pycolmap', max_num_keypoints=1024).eval().cuda()
        self.matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()
        self.feat0 = self.extractor.extract(numpy_image_to_torch(rgb_ref))
        self.seg_ref = seg_ref

    def image_callback(self, msg):
        with self.lock:
            self.live_rgb_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.condition.notify_all()

    def observe(self):
        with self.lock:
            self.live_rgb_image = None
            collected = self.condition.wait_for(lambda: self.live_rgb_image is not None, timeout=1)
            
            if not collected:
                rospy.logwarn("Timeout occurred while waiting for image.")
                raise NotImplementedError

            live_rgb = self.live_rgb_image
            self.live_rgb_image = None
        
        return self.compute_match(live_rgb)

    def compute_match(self, live_rgb, filter_seg=True):
        feats1 = self.extractor.extract(numpy_image_to_torch(live_rgb))
        matches01 = self.matcher({'image0': self.feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [self.feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
        mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)
        
        if filter_seg:
            # Convert to integer coordinates for segmentation lookup
            coords = mkpts_0.astype(int)
            # Get segmentation values for corresponding coordinates
            seg_values = self.seg_ref[coords[:, 1], coords[:, 0]]
            # Filter points where segment values are True
            mask = seg_values

            mkpts_0 = mkpts_0[mask]
            mkpts_1 = mkpts_1[mask]
                    
        return mkpts_0, mkpts_1