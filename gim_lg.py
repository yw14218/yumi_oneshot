import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F
from os.path import join
from PIL import Image

from gim.gluefactory.superpoint import SuperPoint
from gim.gluefactory.models.matchers.lightglue import LightGlue

class GimMatcher:
    def __init__(self, img_path0, checkpoint='gim_lightglue_100h.ckpt'):
        self.img_path0 = img_path0
        self.checkpoints_path = join('gim/weights', checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.detector = SuperPoint({
            'max_num_keypoints': 2048,
            'force_num_keypoints': False,
            'detection_threshold': 0.0,
            'nms_radius': 3,
            'trainable': False,
        }).to(self.device)
        
        self.model = LightGlue({
            'filter_threshold': 0.0,
            'flash': False,
            'checkpointed': True,
        }).to(self.device)
        
        self.load_weights()
        
        # Preprocess image0 once
        self.image0, self.gray0, self.scale0, self.size0 = self.preprocess_cv_image(img_path0)
        
    def load_weights(self):
        state_dict = torch.load(self.checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        
        # Load SuperPoint weights
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict.pop(k)
            if k.startswith('superpoint.'):
                state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
        self.detector.load_state_dict(state_dict)
        
        # Load LightGlue weights
        state_dict = torch.load(self.checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        
        # Set models to evaluation mode
        self.detector.eval()
        self.model.eval()

    def read_image(self, path, grayscale=False):
        if grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(path), mode)
        if image is None:
            raise ValueError(f'Cannot read image {path}.')
        if not grayscale and len(image.shape) == 3:
            image = image[:, :, ::-1]  # BGR to RGB
        return image

    def preprocess(self, image, grayscale=False, resize_max=None, dfactor=8):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])

        if resize_max:
            scale = resize_max / max(size)
            if scale < 1.0:
                size_new = tuple(int(round(x * scale)) for x in size)
                image = self.resize_image(image, size_new, 'cv2_area')
                scale = np.array(size) / np.array(size_new)

        if grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()

        # Assure that the size is divisible by dfactor
        size_new = tuple(map(lambda x: int(x // dfactor * dfactor), image.shape[-2:]))
        image = F.resize(image, size=size_new)
        scale = np.array(size) / np.array(size_new)[::-1]
        return image, scale

    def resize_image(self, image, size, interp):
        assert interp.startswith('cv2_')
        interp = getattr(cv2, 'INTER_' + interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
        return resized

    def preprocess_cv_image(self, image):
        processed_image, scale = self.preprocess(image)
        processed_image = processed_image.to(self.device)[None]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = self.preprocess(gray_image, grayscale=True)[0]
        gray_image = gray_image.to(self.device)[None]
        scale = torch.tensor(scale).to(self.device)[None]
        size = torch.tensor(gray_image.shape[-2:][::-1])[None]

        return processed_image, gray_image, scale, size

    def match_images(self, image1):
        # Preprocess image1
        image1, gray1, scale1, size1 = self.preprocess_cv_image(image1)

        data = dict(
            color0=self.image0, color1=image1,
            image0=self.image0, image1=image1,
            gray0=self.gray0, gray1=gray1,
            size0=self.size0, size1=size1,
            scale0=self.scale0, scale1=scale1
        )

        pred = {}
        pred.update({k + '0': v for k, v in self.detector({
            "image": data["gray0"],
            "image_size": data["size0"],
        }).items()})
        pred.update({k + '1': v for k, v in self.detector({
            "image": data["gray1"],
            "image_size": data["size1"],
        }).items()})
        pred.update(self.model({**pred, **data,
                                **{'resize0': data['size0'], 'resize1': data['size1']}}))

        kpts0 = torch.cat([kp * s for kp, s in zip(pred['keypoints0'], data['scale0'][:, None])])
        kpts1 = torch.cat([kp * s for kp, s in zip(pred['keypoints1'], data['scale1'][:, None])])
        m_bids = torch.nonzero(pred['keypoints0'].sum(dim=2) > -1)[:, 0]
        matches = pred['matches']
        bs = data['image0'].size(0)
        kpts0 = torch.cat([kpts0[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        kpts1 = torch.cat([kpts1[m_bids == b_id][matches[b_id][..., 1]] for b_id in range(bs)])
        # b_ids = torch.cat([m_bids[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        # mconf = torch.cat(pred['scores'])

        return kpts0.cpu().numpy(), kpts1.cpu().numpy()

# Example usage
img_path0 = "experiments/pencile_sharpener/demo_wrist_rgb.png"
img_path1 = "experiments/pencile_sharpener/demo_wrist_rgb.png" 
image0 = cv2.imread(img_path0)[...,::-1]  
image1 = cv2.imread(img_path1)[...,::-1]  
matcher = GimMatcher(image0)
homography_matrix = matcher.match_images(image1)
print(homography_matrix)

