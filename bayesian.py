import rospy
import torch
import cv2
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from yumi_moveit_utils import get_curent_T_left
from scipy.spatial.transform import Rotation
from trajectory_utils import translation_from_matrix, quaternion_from_matrix
from ikSolver import IKSolver
import poselib

class BayesianController:
    def __init__(self, demo_image_path, stable_point, batch_size=10):
        rospy.init_node('yumi_joint_state_publisher', anonymous=True)
        
        self.bridge = CvBridge()
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        self.stable_point = stable_point
        self.demo_rgb = cv2.imread(demo_image_path)[..., ::-1].copy()
        self.publisher_right = rospy.Publisher("/yumi/joint_traj_pos_controller_l/command", JointTrajectory, queue_size=10)
        self.starting_eef_pose = get_curent_T_left()
        self.batch_size = batch_size
        self.ik_solver_left = IKSolver(group_name="left_arm", ik_link_name="gripper_l_base")
        
        self.optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds={'delta_x': (2, 4), 'delta_y': (-3, 3), 'delta_yaw': (-180, 180)},
            random_state=1,
        )
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    
    def compute_transformation_matrix(self, delta_x, delta_y, delta_yaw):
        rotation = Rotation.from_euler('z', delta_yaw, degrees=True).as_matrix()
        T_delta_eef = np.eye(4)
        T_delta_eef[:3, :3] = rotation
        T_delta_eef[0, 3] = delta_x
        T_delta_eef[1, 3] = delta_y
        return T_delta_eef

    def move_eef(self, delta_x, delta_y, delta_yaw):
        T_delta_eef = self.compute_transformation_matrix(delta_x, delta_y, delta_yaw)
        new_eef_pose = get_curent_T_left() @ T_delta_eef
        t = translation_from_matrix(new_eef_pose)
        q = quaternion_from_matrix(new_eef_pose)
        ik_left = self.ik_solver_left.get_ik(new_eef_pose).solution.joint_state.position
        target_joints = ik_left[:7]
        
        msg = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = target_joints
        msg.points.append(point)
        self.publisher_right.publish(msg)

        rospy.sleep(0.1)

    def observe(self):
        predictions = []
        for _ in range(self.batch_size):
            msg = rospy.wait_for_message("d405/color/image_rect_raw", Image)
            live_rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            mkpts_0, mkpts_1 = self.xfeat.match_xfeat_star(self.demo_rgb, live_rgb, top_k=4096)
            H = poselib.estimate_homography(mkpts_0, mkpts_1)[0]
            key_point_demo = np.array(self.stable_point, dtype=np.float32).reshape(-1, 1, 2)
            key_point_live_hom = cv2.perspectiveTransform(key_point_demo, H)
            x, y = key_point_live_hom[0][0]
            predictions.append((x, y))
        
        return np.array(predictions)

    def black_box_function(self, delta_x, delta_y, delta_yaw):
        self.move_eef(delta_x, delta_y, delta_yaw)
        predictions_np = self.observe()
        mean = np.mean(predictions_np, axis=0)
        variance = np.var(predictions_np, axis=0)
        rospy.loginfo(f"mean: {mean}, variance: {variance}")
        
        return -np.sum(variance)

    def optimize(self, iterations=100):
        for _ in range(iterations):
            next_point = self.optimizer.suggest(self.utility)
            target = self.black_box_function(**next_point)
            self.optimizer.register(params=next_point, target=target)
            print(target, next_point)
        print(self.optimizer.max)

if __name__ == "__main__":
    demo_image_path = "/path/to/demo_wrist_rgb.png"
    stable_point = [499.86090324, 103.8931002]

    yumi_optimizer = BayesianController(demo_image_path, stable_point)
    yumi_optimizer.optimize(100)
