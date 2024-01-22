from abc import abstractclassmethod

import torch
import torch.nn as nn

ModelRotPrediction = torch.Tensor
PredictedRotation = float


class RotationPredictorBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractclassmethod
    def load_weights(self):
        pass

    @abstractclassmethod
    def predict_rotation(self) -> ModelRotPrediction:
        pass

    @abstractclassmethod
    def prediction_to_radians(self, pred: ModelRotPrediction) -> PredictedRotation:
        pass

    @abstractclassmethod
    def forward(self, data: dict):
        pass
