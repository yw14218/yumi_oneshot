import abc
import rospy
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Lock, Condition
from camera_utils import d405_rgb_topic_name, d405_depth_topic_name
from trajectory_utils import pose_inv
from moveit_utils.cartesian_control import YuMiLeftArmCartesianController
from lightglue import LightGlue, SIFT, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd
from mini_dust3r.api import inferece_dust3r
from mini_dust3r.model import AsymmetricCroCo3DStereo

class CartesianVisualServoer(abc.ABC):
    def __init__(self, use_depth=False):        
        self.bridge = CvBridge()
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.cartesian_controller = YuMiLeftArmCartesianController()
        self.rgb_subscriber = rospy.Subscriber(d405_rgb_topic_name, Image, self.rgb_image_callback)

        self.images = {
            "rgb": None,
            "depth": None
        }
        self.use_depth = use_depth

        if self.use_depth:
            self.depth_subscriber = rospy.Subscriber(d405_depth_topic_name, Image, self.depth_image_callback)

    def rgb_image_callback(self, msg):
        with self.lock:
            try:
                self.images["rgb"] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                self.condition.notify_all()
            except Exception as e:
                rospy.logerr(f"Error in rgb_image_callback: {e}")

    def depth_image_callback(self, msg):
        with self.lock:
            try:
                self.images["depth"] = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.condition.notify_all()
            except Exception as e:
                rospy.logerr(f"Error in depth_image_callback: {e}")

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
    
    def run(self):
        """Run the visual servoing loop."""
        pass

class LightGlueVisualServoer(CartesianVisualServoer):
    def __init__(self, rgb_ref, seg_ref, use_depth=False, features='sift'):
        super().__init__(use_depth=use_depth)

        if features == 'sift':
            self.extractor_sift = SIFT(backend='pycolmap', max_num_keypoints=1024).eval().cuda()
            self.matcher_sift = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()
            self.feats0_sift = self.extractor_sift.extract(numpy_image_to_torch(rgb_ref))
        elif features == 'superpoint':
            self.extractor_sp = SuperPoint(max_num_keypoints=1024).eval().cuda()
            self.matcher_sp = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda() 
            self.feats0_sp = self.extractor_sp.extract(numpy_image_to_torch(rgb_ref))
        else:
            raise NotImplementedError
        
        self.features = features
        self.seg_ref = seg_ref

    def match_lightglue(self, filter_seg=True):
        live_rgb, live_depth = self.observe()

        if live_rgb is None:
            raise RuntimeError("No image received. Please check the camera and topics.")

        if self.features == 'sift':
            feats1 = self.extractor_sift.extract(numpy_image_to_torch(live_rgb))
            matches01 = self.matcher_sift({'image0': self.feats0_sift, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [self.feats0_sift, feats1, matches01]]
        elif self.features == 'superpoint':
            feats1 = self.extractor_sp.extract(numpy_image_to_torch(live_rgb))
            matches01 = self.matcher_sp({'image0': self.feats0_sp, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [self.feats0_sp, feats1, matches01]]

        matches, scores = matches01['matches'], matches01['scores']
        mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
        mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

        if matches.shape[0] == 0:
            return None, None, None

        if filter_seg:
            coords = mkpts_0.astype(int)
            mask = self.seg_ref[coords[:, 1], coords[:, 0]].astype(bool)
            mkpts_0 = mkpts_0[mask]
            mkpts_1 = mkpts_1[mask]
            valid_indices = np.where(mask)[0] 
            scores = scores[valid_indices]

        scores = scores.detach().cpu().numpy()[..., None]
        mkpts_scores_0 = np.concatenate((mkpts_0, scores), axis=1)
        mkpts_scores_1 = np.concatenate((mkpts_1, scores), axis=1)

        return mkpts_scores_0, mkpts_scores_1, live_depth

class Dust3RisualServoer(CartesianVisualServoer):
    def __init__(self, rgb_ref, use_depth=False):
        super().__init__(use_depth=use_depth)

        self.rgb_ref = rgb_ref
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(device)

    def estimate_rel_pose(self):
        live_rgb, live_depth = self.observe()
        res = inferece_dust3r(
            image_dir_or_list=[self.rgb_ref, live_rgb],
            model=self.model,
            device="cuda",
            batch_size=1
        )
        T_0, T_1 = res[0], res[1]
        if np.allclose(T_1, np.eye(4)):
            T = T_0
        else:
            T = pose_inv(T_1)
        
        return T

if __name__ == "__main__":
    import cv2
    rospy.init_node('sift_lightglue_visual_servoer', anonymous=True)

    dir = "experiments/scissor"

    # Load the reference RGB image and segmentation mask
    rgb_ref = cv2.imread(f"{dir}/demo_wrist_rgb.png")[...,::-1].copy()
    seg_ref = cv2.imread(f"{dir}/demo_wrist_seg.png", cv2.IMREAD_GRAYSCALE).astype(bool)

    # Initialize the visual servoer with reference images
    lgvs = LightGlueVisualServoer(rgb_ref, seg_ref) 

    try:
        while not rospy.is_shutdown():
            mkpts_scores_0, mkpts_scores_1, live_depth = lgvs.match_lightglue(filter_seg=True)

            rospy.loginfo(f"Matched keypoints: {mkpts_scores_0, mkpts_scores_1}")
            rospy.sleep(1)  # Adjust sleep time as needed
    except rospy.ROSInterruptException:
        pass