import numpy as np
import torch
import torch.nn as nn

from direct.pointnet2_utils import PointNetSetAbstractionMsg, \
    PointNetSetAbstraction
from direct.rotation_predictor import RotationPredictor

ModelRotPrediction = torch.Tensor
PredictedRotation = float
PointCloud = torch.Tensor


class PointNet2_Encoder_v2(nn.Module):
    def __init__(self, output_dim, data_dim=0):
        super(PointNet2_Encoder_v2, self).__init__()
        self.data_present = data_dim > 0
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], data_dim,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.mlp1 = nn.Sequential(
            nn.Linear(1027, 512),
            # nn.BatchNorm1d(512),
            nn.InstanceNorm1d(512),
            nn.Dropout(0.2),
            # nn.LeakyReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.InstanceNorm1d(256),
            nn.Dropout(0.2),
            # nn.LeakyReLU(inplace=True)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(256, output_dim),
            # nn.BatchNorm1d(output_dim),
            nn.InstanceNorm1d(output_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, xyz):
        bs, _, _ = xyz.shape
        if self.data_present:
            data = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            data = None
        l1_xyz, l1_points = self.sa1(xyz, data)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l3_points = l3_points.view(bs, 1024)
        l3_xyz = l3_xyz.view(bs, 3)
        out = torch.cat((l3_xyz, l3_points), dim=1)

        out = self.mlp1(out)
        out = self.mlp2(out)
        out = self.mlp3(out)
        # x = F.log_softmax(x, -1)

        return out


class RotationClassificator(RotationPredictor):

    def __init__(self, n_classes: int = 90, initialise_pretrained: bool = False) -> None:
        super(RotationClassificator, self).__init__()

        self.encoder = PointNet2_Encoder_v2(output_dim=256, data_dim=6)
        self.fusion = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                    # nn.BatchNorm1d(256),
                                    nn.InstanceNorm1d(256),
                                    nn.LeakyReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                  #   nn.BatchNorm1d(128),
                                  nn.InstanceNorm1d(128),
                                  nn.LeakyReLU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=128, out_features=n_classes, bias=True),
                                  nn.Identity(inplace=True))

        self.n_classes = n_classes
        # self.weights_dir = "class_best_real_val.ckpt"
        self.weights_dir = "last_class.ckpt"

        if initialise_pretrained:
            self.load_weights()

    def predict_rotation(self, pcd0: PointCloud, pcd1: PointCloud) -> ModelRotPrediction:
        encoded_live = self.encoder(pcd0)
        encoded_bottleneck = self.encoder(pcd1)

        out = torch.concat((encoded_live, encoded_bottleneck), dim=1)
        out = self.fusion(out)
        out = self.mlp2(out)
        out = self.mlp3(out)

        return out.cpu()

    def prediction_to_radians(self, pred: ModelRotPrediction) -> PredictedRotation:
        bin_size = (np.pi / 2) / self.n_classes
        rot_pred = (torch.argmax(pred.detach(), dim=1) + 0.5) * bin_size - (np.pi / 4)
        return rot_pred.cpu().numpy()


if __name__ == "__main__":
    rand_live = np.random.rand(9, 1000).astype(np.float32)
    rand_bottle = np.random.rand(9, 1000).astype(np.float32)

    batch = {
        "pointcloud_0": rand_live,
        "pointcloud_1": rand_bottle,
    }

    model = RotationClassificator()
    model.load_weights()
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(np.rad2deg(out))
