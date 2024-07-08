from PIL import Image
import torch
import cv2
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from rotation_steerers.steerers import DiscreteSteerer, ContinuousSteerer
from rotation_steerers.matchers.max_similarity import MaxSimilarityMatcher, ContinuousMaxSimilarityMatcher

im_A_path = "rotation-steerers/example_images/im_A_rot.jpg"
im_B_path = "rotation-steerers/example_images/im_B.jpg"
im_A = Image.open(im_A_path)
im_B = Image.open(im_B_path)
w_A, h_A = im_A.size
w_B, h_B = im_B.size

# Detection of keypoints (as for ordinary DeDoDe)
detector = dedode_detector_L(weights=torch.load("rotation-steerers/model_weights/dedode_detector_L.pth"))
detections_A = detector.detect_from_image(im_A, num_keypoints = 10_000)
keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]
detections_B = detector.detect_from_image(im_B, num_keypoints = 10_000)
keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]

# C8-steering with discretized steerer (recommended)
descriptor = dedode_descriptor_B(weights=torch.load("rotation-steerers/model_weights/B_SO2_Spread_descriptor_setting_B.pth"))
steerer_order = 8
steerer = DiscreteSteerer(
    generator=torch.matrix_exp(
        (2 * 3.14159 / steerer_order)
        * torch.load("rotation-steerers/model_weights/B_SO2_Spread_steerer_setting_B.pth")
    )
)
matcher = MaxSimilarityMatcher(steerer=steerer, steerer_order=steerer_order)

# Describe keypoints and match descriptions (API as in DeDoDe)
descriptions_A = descriptor.describe_keypoints_from_image(im_A, keypoints_A)["descriptions"]
descriptions_B = descriptor.describe_keypoints_from_image(im_B, keypoints_B)["descriptions"]

matches_A, matches_B, batch_ids = matcher.match(
    keypoints_A, descriptions_A,
    keypoints_B, descriptions_B,
    P_A = P_A, P_B = P_B,
    normalize = True, inv_temp=20, threshold = 0.01
)
matches_A, matches_B = matcher.to_pixel_coords(
    matches_A, matches_B, 
    h_A, w_A, h_B, w_B,
)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torch
    import numpy as np
    from PIL import Image
    import cv2
    import matplotlib.pyplot as plt

# Helper function for drawing matches
def draw_img_match(img1, img2, mkpts1, mkpts2, reverse_pair=True):
    if isinstance(img1, torch.Tensor):
        img1 = im_tensor_to_np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = im_tensor_to_np(img2)
    if isinstance(mkpts1, torch.Tensor):
        mkpts1 = mkpts1.detach().cpu().numpy()
    if isinstance(mkpts2, torch.Tensor):
        mkpts2 = mkpts2.detach().cpu().numpy()

    if isinstance(img1, np.ndarray):
        img1 = np.uint8(255 * img1)
    else:
        img1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    if isinstance(img2, np.ndarray):
        img2 = np.uint8(255 * img2)
    else:
        img2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)

    if reverse_pair:
        img1, img2, = img2, img1
        mkpts1, mkpts2 = mkpts2, mkpts1

    img = cv2.drawMatches(
        img1=img1,
        keypoints1=[cv2.KeyPoint(x=x, y=y, size=2) for x, y in mkpts1],
        img2=img2,
        keypoints2=[cv2.KeyPoint(x=x, y=y, size=2) for x, y in mkpts2],
        matches1to2=[cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=-1.) for i in range(len(mkpts1))],
        matchesThickness=2,
        outImg=None,
    )
    plt.imshow(img)

def im_tensor_to_np(x):
    return x[0].permute(1, 2, 0).detach().cpu().numpy()

print(type(matches_A))
print(matches_A)
H, _ = cv2.findHomography(matches_A.cpu().numpy(), matches_B.cpu().numpy(), cv2.USAC_MAGSAC)
print(H)

plt.figure(figsize=(10,5), dpi=80)
draw_img_match(im_A_path, im_B_path, matches_A[:100], matches_B[:100])
plt.axis('off')
plt.show()