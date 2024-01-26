import os
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch.nn.parameter import Parameter

from rotation_predictor import RotationPredictorBase


class ProcessedData(TypedDict):
    pointcloud_0: np.ndarray
    pointcloud_1: np.ndarray


ModelRotPrediction = torch.Tensor
PredictedRotation = float
PointCloud = torch.Tensor


class RotationPredictor(RotationPredictorBase):

    def __init__(self) -> None:
        super().__init__()
        self.weights_dir: str = None

    def load_weights(self):
        print("\n[POSE ESTIMATOR] Loading pretrained weights ...")
        ckpt_dir = os.path.join(Path(__file__).parent, 'weights', self.weights_dir)
        state_dict = torch.load(ckpt_dir, map_location='cpu')['state_dict']
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.split(".", 1)[1]
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        print("[POSE ESTIMATOR] Done.\n")

    def forward(self, data: ProcessedData):
        pcd0: PointCloud = torch.tensor(data["pointcloud_0"]).unsqueeze(0)  # .to(device=self.device)
        pcd1: PointCloud = torch.tensor(data["pointcloud_1"]).unsqueeze(0)  # .to(device=self.device)
        with torch.no_grad():
            model_pred = self.predict_rotation(pcd0, pcd1)
        predicted_rot = self.prediction_to_radians(model_pred)
        return predicted_rot
