from direct.rotation_classificator import RotationClassificator
from direct.rotation_regressor import RotationRegressor
from pose_predictor import PosePredictor


class PointentPoseClassifier(PosePredictor):

    def __init__(self,
                 device: str = "cuda:0",
                 filter_pointcloud: bool = True,
                 filter_outliers_o3d: bool = False,
                 n_points: int = 2048) -> None:
        super().__init__(device, filter_pointcloud, filter_outliers_o3d, n_points)
        self.rot_model = RotationClassificator()
        self._initialise_models()


class PointentPoseRegressor(PosePredictor):

    def __init__(self,
                 device: str = "cuda:0",
                 filter_pointcloud: bool = True,
                 filter_outliers_o3d: bool = False,
                 n_points: int = 2048) -> None:
        super().__init__(device, filter_pointcloud, filter_outliers_o3d, n_points)
        self.rot_model = RotationRegressor()
        self._initialise_models()
