import torch
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image

DIR = 'experiments/scissor'
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)

rgb_message = rospy.wait_for_message(f"/d415/color/image_raw", ImageMsg, timeout=5)
live_rgb = ros_numpy.numpify(rgb_message)
demo_rbg = Image.open(f"{DIR}/demo_head_rgb_seg.png")
mkpts_0, mkpts_1 = xfeat.match_xfeat_star(im1, im2, top_k = 4096)