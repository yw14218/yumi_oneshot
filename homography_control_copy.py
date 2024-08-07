import rospy
import torch
import cv2
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction, SequentialDomainReductionTransformer
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation
from trajectory_utils import *
from ikSolver import IKSolver
import poselib
from poseEstimation import PoseEstimation
import json
import time
from threading import Event, Lock
from camera_utils import convert_from_uvd, d405_K as K, d405_T_C_EEF as T_C_EEF
from matplotlib import pyplot as plt
from matplotlib import gridspec
from lightglue import LightGlue, SuperPoint, match_pair
from xfeat_listener import numpy_image_to_torch, decompose_homography
from geometry_msgs.msg import geometry_msgs
import yumi_moveit_utils as yumi
import matplotlib.pyplot as plt
from vis import *

class ImageListener:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.live_rgb_batch = []
        self.bridge = CvBridge() 
        self.image_event = Event()
        self.lock = Lock()
        self.subscriber = rospy.Subscriber("d405/color/image_rect_raw", Image, self.image_callback)

    def image_callback(self, msg):
        with self.lock:
            if len(self.live_rgb_batch) < self.batch_size:
                self.live_rgb_batch.append(self.bridge.imgmsg_to_cv2(msg, "rgb8"))
                if len(self.live_rgb_batch) == self.batch_size:
                    self.image_event.set()

    def observe(self):
        with self.lock:
            self.live_rgb_batch = []
        self.image_event.clear()
        
        # Wait for batch_size images to be collected or timeout
        collected = self.image_event.wait(1)
        
        if not collected:
            rospy.logwarn("Timeout occurred while waiting for images.")
            raise NotImplementedError

        with self.lock:
            for live_rgb in self.live_rgb_batch:
                feats0, feats1, matches01 = match_pair(extractor, matcher, demo_rgb_cuda, numpy_image_to_torch(live_rgb))
                matches = matches01['matches']  # indices with shape (K,2)
                mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
                mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)

                try:
                    H, _ = cv2.findHomography(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 5.0)
                except cv2.error:
                    return None
            return H


def get_current_stab_3d(T_EEF_World):
    stab_point3d = pose_inv(T_EEF_World @ T_C_EEF) @ T_stab_pose @ T_GRIP_EEF
    # Project the 3D point onto the image plane
    return np.dot(K, stab_point3d[:3, 3])

def move_eef(new_eef_pose):

    ik_left = ik_solver_left.get_ik(new_eef_pose).solution.joint_state.position
    target_joints = ik_left[:7]
    
    msg = JointTrajectory()
    point = JointTrajectoryPoint()
    point.positions = target_joints
    point.time_from_start = rospy.Duration(0.05)
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = [f"yumi_joint_{i}_l" for i in [1, 2, 7, 3, 4, 5, 6]]
    msg.points.append(point)
    publisher_left.publish(msg)

rospy.init_node('yumi_bayesian_controller', anonymous=True)

publisher_left = rospy.Publisher("/yumi/joint_traj_pos_controller_l/command", JointTrajectory, queue_size=10)
ik_solver_left = IKSolver(group_name="left_arm", ik_link_name="gripper_l_base")
imageListener = ImageListener(batch_size=3)

DIR = "experiments/pencile_sharpener"
OBJ = "blue pencile sharpener"
# from experiments.pencile_sharpener.experiment import SharpenerExperiment as Experiment

# DIR = "experiments/wood"
# OBJ = "wooden stand"
from experiments.wood.experiment import WoodExperiment as Experiment

T_GRIP_EEF = create_homogeneous_matrix([-0.07, 0.08, 0.2], [0, 0, 0, 1])

pose_estimator = PoseEstimation(
    dir=DIR,
    text_prompt=OBJ,
    visualize=False
)

file_name = f"{DIR}/demo_bottlenecks.json"
with open(file_name) as f:
    dbn = json.load(f)
demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])

bottleneck_left = demo_waypoints[0].tolist()
bottleneck_right = demo_waypoints[1].tolist()
stab_pose = dbn["bottleneck_left"]
T_bottleneck_left = create_homogeneous_matrix(bottleneck_left[:3], bottleneck_left[3:])
T_stab_pose = create_homogeneous_matrix(stab_pose[:3], stab_pose[3:])
stab_3d_cam = get_current_stab_3d(T_EEF_World=T_bottleneck_left)

# Normalize the coordinates to get the 2D image point
stab_point_2D = stab_3d_cam[:2] / stab_3d_cam[2]
stab_point_2D_np = np.array(stab_point_2D, dtype=np.float32).reshape(-1, 1, 2)
# xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)   

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher
demo_rgb = cv2.imread(f"{DIR}/demo_wrist_rgb.png")[..., ::-1].copy() 
demo_seg = cv2.imread(f"{DIR}/demo_wrist_seg.png")
demo_rgb_cuda = numpy_image_to_torch(demo_rgb * demo_seg.astype(bool))
template_mask = cv2.threshold(demo_seg, 127, 255, cv2.THRESH_BINARY)[1]

yumi.init_Moveit()

# yumi.reset_init()
# bottleneck_left[2] += 0.1
yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))
user_input = input("Continue? (yes/no): ").lower()
if user_input == "ready":
    pass

# yumi.gripper_effort(yumi.LEFT, 20)

# diff_xyz, diff_rpy = pose_estimator.decouple_run(output_path=f"{DIR}/", camera_prefix="d415")
# # rospy.loginfo(f"Diff xyz is {diff_xyz}, diff rpy is {diff_rpy}")
# bottleneck_left_new = bottleneck_left.copy()
# bottleneck_left_new[0] += diff_xyz[0]
# bottleneck_left_new[1] += diff_xyz[1]
# # bottleneck_left_new[2] += 0.05
# bottleneck_euler = euler_from_matrix(T_bottleneck_left)
# bottleneck_euler[-1] += diff_rpy[-1]
# yumi.plan_left_arm(yumi.create_pose_euler(*bottleneck_left_new[:3], *bottleneck_euler))
yumi.plan_left_arm(yumi.create_pose(*bottleneck_left))


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    
def iterative_learning_control(m, K, max_iterations=200, threshold=0.1):
    current_pose = yumi.get_current_pose(yumi.LEFT).pose
    current_rpy = euler_from_quat([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
    control_input_x = 0
    control_input_y = 0
    errors = []
    pid_x = PIDController(Kp=0.15, Ki=0.0, Kd=0.05)
    pid_y = PIDController(Kp=0.15, Ki=0.0, Kd=0.05)
    pid_z = PIDController(Kp=0.15, Ki=0.0, Kd=0.05)

    pid_theta = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
    trajectory = []
    I = np.eye(3)
    cap_value =  0.001
    for iteration in range(max_iterations):
        # Capture current image and detect projection pixel
        H = imageListener.observe()

        if H is None:
            continue
        dx, dy, dz = (I - H) @ m

        print(dx, dy, dz)
        drz = (H - H.T)[1, 0]
        # Calculate pixel error
        
        error = np.linalg.norm(H - I, ord=1)
        
        # Check if error is within threshold
        if (error < 1) or iteration == max_iterations - 1:
            # T_delta_world = yumi.get_curent_T_left() @ pose_inv(T_bottleneck_left)
            # live_waypoints = apply_transformation_to_waypoints(demo_waypoints, T_delta_world, project3D=True)
            # Experiment.replay(live_waypoints)
            break

        errors.append(error)

        control_input_x = pid_x.update(dx)
        control_input_y = pid_y.update(dy)
        control_input_z = pid_z.update(dz)
        control_input_z_rot = pid_theta.update(drz)

        if abs(control_input_x) >= cap_value:
            control_input_x = cap_value if control_input_x > 0 else -cap_value
        if abs(control_input_y) >= cap_value:
            control_input_y = cap_value if control_input_y > 0 else -cap_value
        # if abs(control_input_z) >= cap_value:
        #     control_input_z = cap_value if control_input_z > 0 else -cap_value
            
        if abs(control_input_z_rot) >= 0.04:
            control_input_z_rot = 0.04 if control_input_z_rot > 0 else -0.04


        rospy.loginfo(f"Step {iteration + 1}, Error is : {error:.4g}, delta_x: {control_input_x:.4g}, delta_y: {control_input_y:.4g}, delta_yaw: {np.degrees(control_input_z_rot)}, delta_z: {control_input_z:.4g}")
    
        # Move robot by the updated control input
        current_pose.position.x += control_input_x
        current_pose.position.y -= control_input_y
        # current_pose.position.z -= control_input_z
        current_rpy[-1] -= control_input_z_rot
        
        trajectory.append([current_pose.position.x, current_pose.position.y, np.degrees(current_rpy[-1])])

        new_pose = yumi.create_pose_euler(current_pose.position.x, current_pose.position.y, current_pose.position.z, current_rpy[0], current_rpy[1], current_rpy[2])

        move_eef(new_pose)

    visualize_convergence_on_sphere(np.array(trajectory))
    
    return current_pose

iterative_learning_control(m=stab_3d_cam, K=K)

rospy.spin()