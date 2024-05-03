import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from PIL import Image
import copy
from sensor_msgs.msg import Image as ImageMsg
from PoseEst.direct.preprocessor import Preprocessor, pose_inv, SceneData
from langSAM import LangSAMProcessor


class PoseEstimation:
    def __init__(self, text_prompt, demo_rgb_path, demo_depth_path, demo_mask_path, intrinsics_path, T_WC_path):
        self.langSAMProcessor = LangSAMProcessor(text_prompt=text_prompt)
        self.demo_rgb_path = demo_rgb_path
        self.demo_depth_path = demo_depth_path
        self.demo_mask_path = demo_mask_path
        self.intrinsics_path = intrinsics_path
        self.T_WC_path = T_WC_path

    def get_live_data(self, timeout=5):
        """
        Retrieve live RGB and depth data with a specified timeout.

        Args:
            timeout (int): The number of seconds to wait for a message before timing out.

        Returns:
            tuple: A tuple containing the live RGB and depth data as numpy arrays.

        Raises:
            rospy.exceptions.ROSException: If the message is not received within the timeout.
        """
        try:
            rgb_message = rospy.wait_for_message("/d415/color/image_raw", ImageMsg, timeout=timeout)
            depth_message = rospy.wait_for_message("/d415/aligned_depth_to_color/image_raw", ImageMsg, timeout=timeout)
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"Data acquisition timed out: {e}")
            raise

        live_rgb = ros_numpy.numpify(rgb_message)
        live_depth = ros_numpy.numpify(depth_message)
        
        assert live_rgb.shape[0] == live_depth.shape[0] == 720

        return live_rgb, live_depth

    def inference_and_save(self, folder_path):
        live_rgb, live_depth = self.get_live_data()
        rgb_image = Image.fromarray(live_rgb)
        depth_image = Image.fromarray(live_depth)

        # Ensure mask_np is correctly generated and not None before proceeding
        mask_np = self.langSAMProcessor.inference(live_rgb, single_mask=True, visualize_info=True)
        if mask_np is None:
            raise ValueError("No mask returned from inference.")

        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))

        import os
        self.live_rgb_path = os.path.join(folder_path, "live__head_rgb.png")
        self.live_depth_path = os.path.join(folder_path, "live_head_depth.png")
        self.live_mask_path = os.path.join(folder_path, "live_head_seg.png")
        rgb_image.save(self.live_rgb_path)
        depth_image.save(self.live_depth_path)
        mask_image.save(self.live_mask_path)

        return rgb_image, depth_image, mask_image


    def process_data(self, rgb_image, depth_image, mask_image):


        demo_rgb = np.array(Image.open(self.demo_rgb_path))
        demo_depth = np.array(Image.open(self.demo_depth_path))
        demo_mask = np.array(Image.open(self.demo_mask_path))
        
        live_rgb = np.array(rgb_image)
        live_depth = np.array(depth_image)
        live_mask = np.array(mask_image)
        intrinsics = np.load(self.intrinsics_path)

        data = SceneData(
            image_0=demo_rgb,
            image_1=live_rgb,
            depth_0=demo_depth,
            depth_1=live_depth,
            seg_0=demo_mask,
            seg_1=live_mask,
            intrinsics_0=intrinsics,
            intrinsics_1=intrinsics,
            T_WC=np.eye(4)  # cam frame
        )

        processor = Preprocessor()
        data.update(processor(data))

        return data

    def estimate_pose(self, data):

        # Function to draw registration results
        def draw_registration_result(source, target, transformation):
            source_temp = copy.deepcopy(source)
            target_temp = copy.deepcopy(target)
            source_temp.transform(transformation)
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])
            o3d.visualization.draw_geometries([source_temp, target_temp])

        pcd0 = o3d.geometry.PointCloud()
        pcd1 = o3d.geometry.PointCloud()

        pcd0.points = o3d.utility.Vector3dVector(data["pc0"][:, :3])
        pcd1.points = o3d.utility.Vector3dVector(data["pc1"][:, :3])


        # o3d.visualization.draw_geometries([pcd0, pcd1])

        # Estimate normals for each point cloud
        pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=40))
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=40))


        # Compute FPFH features
        voxel_size = 0.05  # Set voxel size for downsampling (adjust based on your data)
        source_down = pcd0.voxel_down_sample(voxel_size)
        target_down = pcd1.voxel_down_sample(voxel_size)

        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

        # Global registration using RANSAC
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, mutual_filter=False,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(), 
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), 
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        # Use the result of global registration as the initial transformation for ICP
        trans_init = result.transformation

        # Apply ICP
        threshold = 0.02  # Set a threshold for ICP, this depends on your data
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Get the transformation matrix
        T_delta_cam = reg_p2p.transformation

        # Draw the result
        # draw_registration_result(pcd0, pcd1, T_delta_cam)
        rospy.loginfo(f"ICP Fitness: {reg_p2p.fitness}")
        rospy.loginfo(f"ICP Inlier RMSE: {reg_p2p.inlier_rmse}")
        
        T_WC = np.load("handeye/T_WC_head.npy")
        T_delta_world = T_WC @ T_delta_cam @ pose_inv(T_WC)
        rospy.loginfo("T_delta_world is {0}".format(T_delta_world))

        return T_delta_world

    def run(self, output_path):
        rgb_image, depth_image, mask_image = self.inference_and_save(output_path)
        data = self.process_data(rgb_image, depth_image, mask_image)
        return self.estimate_pose(data)

if __name__ == '__main__':
    rospy.init_node('PoseEstimation', anonymous=True)
    pose_estimator = PoseEstimation(
        text_prompt="lego",
        demo_rgb_path="data/lego_split/demo_rgb.png",
        demo_depth_path="data/lego_split/demo_depth.png",
        demo_mask_path="data/lego_split/demo_mask.png",
        intrinsics_path="handeye/intrinsics_d415.npy",
        T_WC_path="handeye/T_WC_head.npy"
    )
    try:
        T_delta_world = pose_estimator.run(output_path="data/lego_split/")

    except Exception as e:
        print(f"Error: {e}")

