import rospy
import torch
import cv2
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction, SequentialDomainReductionTransformer
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
import yumi_moveit_utils as yumi
from scipy.spatial.transform import Rotation
from trajectory_utils import translation_from_matrix, quaternion_from_matrix, pose_inv, project3D, create_homogeneous_matrix
from ikSolver import IKSolver
import poselib
from poseEstimation import PoseEstimation
import json
import time
from threading import Event, Lock
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF
from matplotlib import pyplot as plt
from matplotlib import gridspec

class BayesianController:
    def __init__(self, demo_image_path, stablizing_pose, batch_size=5):
        
        self.bridge = CvBridge()
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        self.T_stablizing_pose = create_homogeneous_matrix(stablizing_pose[:3], stablizing_pose[3:])
        self.T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.136], [0, 0, 0, 1])
        self.stable_point = [499.86090324, 103.8931002]
        self.demo_rgb = cv2.imread(demo_image_path)[..., ::-1].copy()
        self.demo_rgb_batch = np.tile(self.demo_rgb, (batch_size, 1, 1, 1))
        self.publisher_left = rospy.Publisher("/yumi/joint_traj_pos_controller_l/command", JointTrajectory, queue_size=10)
        self.starting_eef_pose = yumi.get_curent_T_left()
        self.batch_size = batch_size
        self.ik_solver_left = IKSolver(group_name="left_arm", ik_link_name="gripper_l_base")
        self.height = self.starting_eef_pose[2, 3]
        self.descent_rate = 0
        self.step_size = 0.001
        self.delta_x = 0
        self.delta_y = 0
        self.min_variance_and_corresponding_point3d = [-1e6, []]
    
        self.xyz_eef_goal = None
        self.live_rgb_batch = []
        self.image_event = Event()
        self.lock = Lock()
        self.subscriber = rospy.Subscriber("d405/color/image_rect_raw", Image, self.image_callback)
        
        self.optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds={'delta_x': (-0.01, 0.01), 'delta_y': (-0.01, 0.01)},
            random_state=1,
            # bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.5),
            verbose=2
        )
        self.utility = UtilityFunction(kind="ei", xi=1e-4)
    
    def image_callback(self, msg):
        with self.lock:
            if len(self.live_rgb_batch) < self.batch_size:
                self.live_rgb_batch.append(self.bridge.imgmsg_to_cv2(msg, "rgb8"))
                if len(self.live_rgb_batch) == self.batch_size:
                    self.image_event.set()

    def compute_new_stable_point(self):
        point3d = pose_inv(T_C_EEF) @ pose_inv(yumi.get_curent_T_left()) @ self.T_stablizing_pose @ self.T_GRIP_EEF
        point_image_homogeneous = np.dot(K, point3d[:3, 3])
        # Normalize the coordinates to get the 2D image point
        point_2D = point_image_homogeneous[:2] / point_image_homogeneous[2]

        return point_2D

    def compute_transformation_matrix(self, delta_x, delta_y, delta_yaw):
        rotation = Rotation.from_euler('z', delta_yaw, degrees=True).as_matrix()
        T_delta_eef = np.eye(4)
        T_delta_eef[:3, :3] = rotation
        T_delta_eef[0, 3] = delta_x
        T_delta_eef[1, 3] = delta_y
        T_delta_eef[2, 3] = self.descent_rate
        return T_delta_eef

    def move_eef(self, delta_x, delta_y, delta_yaw):
        T_delta_eef = self.compute_transformation_matrix(delta_x, delta_y, delta_yaw)
        T_new_eef_pose = self.starting_eef_pose @ T_delta_eef
        t = translation_from_matrix(T_new_eef_pose).tolist()
        q = quaternion_from_matrix(T_new_eef_pose).tolist()
        new_eef_pose = yumi.create_pose(*[t+q][0])
        ik_left = self.ik_solver_left.get_ik(new_eef_pose).solution.joint_state.position
        target_joints = ik_left[:7]
        
        msg = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = rospy.Duration(0.1)
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = [f"yumi_joint_{i}_l" for i in [1, 2, 7, 3, 4, 5, 6]]
        msg.points.append(point)
        self.publisher_left.publish(msg)
        
    def observe(self, timeout=1):
        predictions = []

        with self.lock:
            self.live_rgb_batch = []
        self.image_event.clear()
        
        # Wait for batch_size images to be collected or timeout
        collected = self.image_event.wait(timeout)
        
        if not collected:
            rospy.logwarn("Timeout occurred while waiting for images.")
            return None

        with self.lock:
            mkpts_list = self.xfeat.match_xfeat_star(self.demo_rgb_batch, np.array(self.live_rgb_batch), top_k=4096)
            for mkpts in mkpts_list:
                mkpts_0_np = mkpts[:, :2].cpu().numpy().reshape(-1, 2)  # Convert tensor to numpy array
                mkpts_1_np = mkpts[:, 2:].cpu().numpy().reshape(-1, 2)  # Convert tensor to numpy array
                H, _ = poselib.estimate_homography(mkpts_0_np, mkpts_1_np)
                key_point_demo = np.array(self.stable_point, dtype=np.float32).reshape(-1, 1, 2)
                key_point_live_hom = cv2.perspectiveTransform(key_point_demo, H)
                x, y = key_point_live_hom[0][0]
                predictions.append((x, y))
            
            return np.array(predictions)

    def black_box_function(self, delta_yaw):
        self.move_eef(self.delta_x, self.delta_y, delta_yaw)
        predictions_np = self.observe()
        mean = np.mean(predictions_np, axis=0)
        variance = np.var(predictions_np, axis=0)
        err = np.linalg.norm(predictions_np - np.array(self.compute_new_stable_point()))
        rospy.loginfo(f"mean: {mean}, variance: {variance}, error: {err}")
        
        return -np.sum(err), mean

    def optimize(self, iterations=100):
        start = time.time()
        for _ in range(iterations):
            if self.height < 0.35:
                break
            next_point = self.optimizer.suggest(self.utility)
            target, prediction = self.black_box_function(**next_point)
            self.optimizer.register(params=next_point, target=target)

            # print(target, next_point)
            self.descent_rate += self.step_size
            self.height -= self.step_size
            
        print(f"*****************Time taken: {time.time() - start}**********************")
        rospy.loginfo(f"Optimal found within {iterations} iterations: {self.optimizer.max}. Corresponding 3d xyz is : {self.min_variance_and_corresponding_point3d[1]}")

        return self.optimizer.max

    def posterior(self, optimizer, x_obs, y_obs, grid):
        optimizer._gp.fit(x_obs, y_obs)

        mu, sigma = optimizer._gp.predict(grid, return_std=True)
        return mu, sigma

    def plot_gp(self, optimizer):
        x = np.linspace(-5, 5, 10000).reshape(-1, 1)
        fig = plt.figure(figsize=(16, 10))
        steps = len(optimizer.space)
        fig.suptitle(
            'Gaussian Process and Utility Function After {} Steps'.format(steps),
            fontdict={'size':30}
        )

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        axis = plt.subplot(gs[0])
        acq = plt.subplot(gs[1])

        x_obs = np.array([[res["params"]["delta_yaw"]] for res in optimizer.res])
        y_obs = np.array([res["target"] for res in optimizer.res])

        mu, sigma = self.posterior(optimizer, x_obs, y_obs, x)
        axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
        axis.plot(x, mu, '--', color='k', label='Prediction')

        axis.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
            alpha=.6, fc='c', ec='None', label='95% confidence interval')

        axis.set_xlim((-2, 10))
        axis.set_ylim((None, None))
        axis.set_ylabel('Cost', fontdict={'size':20})
        axis.set_xlabel('delta_yaw', fontdict={'size':20})

        utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
        utility = utility_function.utility(x, optimizer._gp, 0)
        acq.plot(x, utility, label='Utility Function', color='purple')
        acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
                label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
        acq.set_xlim((-2, 10))
        acq.set_ylim((0, np.max(utility) + 0.5))
        acq.set_ylabel('Utility', fontdict={'size':20})
        acq.set_xlabel('x', fontdict={'size':20})

        axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
        acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

        plt.show()

def get_current_stab_3d(T_EEF_World):
    stab_point3d = pose_inv(T_EEF_World @ T_C_EEF) @ T_stab_pose @ T_GRIP_EEF
    # Project the 3D point onto the image plane
    return np.dot(K, stab_point3d[:3, 3])

if __name__ == "__main__":
    rospy.init_node('yumi_bayesian_controller', anonymous=True)
    DIR = "experiments/wood"
    OBJ = "wood stand"
    T_GRIP_EEF = create_homogeneous_matrix([0, 0, 0.100], [0, 0, 0, 1])
    demo_image_path = f"{DIR}/demo_wrist_rgb.png"

    yumi.init_Moveit()
    
    file_name = f"{DIR}/demo_bottlenecks.json"
    with open(file_name) as f:
        dbn = json.load(f)
    demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

    file_name = f"{DIR}/demo_bottlenecks.json"


    bottleneck_left = demo_waypoints[0].tolist()
    stab_pose = demo_waypoints[3].tolist()
    T_bottleneck_left = create_homogeneous_matrix(bottleneck_left[:3], bottleneck_left[3:])
    T_stab_pose = create_homogeneous_matrix(stab_pose[:3], stab_pose[3:])
    stab_3d_cam = get_current_stab_3d(T_EEF_World=T_bottleneck_left)
    # Normalize the coordinates to get the 2D image point
    stab_point_2D = stab_3d_cam[:2] / stab_3d_cam[2]
    
    yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))

    try:
        pose_estimator = PoseEstimation(
            dir=DIR,
            text_prompt=OBJ,
            visualize=False
        )

        T_delta_cam = pose_estimator.run(output_path=f"{DIR}/", camera_prefix="d405")
        T_camera_eef = np.load(pose_estimator.T_CE_l_path)
        T_new_eef_world = yumi.get_curent_T_left() @ T_camera_eef @ T_delta_cam @ pose_inv(T_camera_eef)
        xyz = translation_from_matrix(T_new_eef_world).tolist()
        quaternion = quaternion_from_matrix(T_new_eef_world).tolist()
        pose_new_eef_world_l = project3D(xyz + quaternion, demo_waypoints[0])
        yumi.plan_left_arm(pose_new_eef_world_l)

        yumi_optimizer = BayesianController(demo_image_path=demo_image_path, stablizing_pose=demo_waypoints[2])
        optimal = yumi_optimizer.optimize(30)
        r = optimal['params']['delta_yaw']
        print(f"Best delta_yaw found: {r}")
        yumi_optimizer.move_eef(0 ,0, r)    
        yumi_optimizer.plot_gp(yumi_optimizer.optimizer)

        # x, y, z = yumi_optimizer.min_variance_and_corresponding_point3d[1]
        # yumi.static_tf_broadcast("world", "goal_world", [x, y, z, 0, 0, 0, 1])
        # ScissorExperiment.replay(live_waypoints=)
        rospy.spin()

    except Exception as e:
        print(f"Error: {e}")