import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import copy
from sensor_msgs.msg import Image as ImageMsg
from PoseEst.direct.preprocessor import Preprocessor, pose_inv, SceneData
from langSAM import LangSAMProcessor
from trajectory_utils import create_homogeneous_matrix

class PoseEstimation:
    def __init__(self, dir, text_prompt, visualize):
        self.langSAMProcessor = LangSAMProcessor(text_prompt=text_prompt)
        self.dir = dir
        self.visualize = visualize
        self.demo_head_rgb_path = f"{self.dir}/demo_head_rgb.png"
        self.demo_head_depth_path = f"{self.dir}/demo_head_depth.png"
        self.demo_head_mask_path = f"{self.dir}/demo_head_seg.png"
        
        self.demo_wrist_rgb_path = f"{self.dir}/demo_wrist_rgb.png"
        self.demo_wrist_depth_path = f"{self.dir}/demo_wrist_depth.png"
        self.demo_wrist_mask_path = f"{self.dir}/demo_wrist_seg.png"

        self.intrinsics_d415_path = "handeye/intrinsics_d405.npy"
        self.T_WC_path = "handeye/T_WC_head.npy"

        self.intrinsics_d405_path = "handeye/intrinsics_d415.npy"
        self.T_CE_l_path = "handeye/T_C_EEF_wrist_l.npy"
        

    def get_live_data(self, camera_prefix, timeout=5):
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
            if camera_prefix == "d415":
                rgb_message = rospy.wait_for_message(f"/d415/color/image_raw", ImageMsg, timeout=timeout)
                depth_message = rospy.wait_for_message(f"/d415/aligned_depth_to_color/image_raw", ImageMsg, timeout=timeout)
            elif camera_prefix == "d405":
                rgb_message = rospy.wait_for_message("d405/color/image_rect_raw", ImageMsg, timeout=timeout)
                depth_message = rospy.wait_for_message("d405/aligned_depth_to_color/image_raw", ImageMsg, timeout=timeout)
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"Data acquisition timed out: {e}")
            raise

        live_rgb = ros_numpy.numpify(rgb_message)
        live_depth = ros_numpy.numpify(depth_message)

        return live_rgb, live_depth

    def inference_and_save(self, camera_prefix, folder_path):
        live_rgb, live_depth = self.get_live_data(camera_prefix)
        rgb_image = Image.fromarray(live_rgb)
        depth_image = Image.fromarray(live_depth)

        # Ensure mask_np is correctly generated and not None before proceeding
        mask_np = self.langSAMProcessor.inference(live_rgb, single_mask=True, visualize_info=self.visualize)
        if mask_np is None:
            raise ValueError("No mask returned from inference.")

        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))

        import os
        self.live_rgb_path = os.path.join(folder_path, f"live_{camera_prefix}_rgb.png")
        self.live_depth_path = os.path.join(folder_path, f"live_{camera_prefix}_depth.png")
        self.live_mask_path = os.path.join(folder_path, f"live_{camera_prefix}_seg.png")
        rgb_image.save(self.live_rgb_path)
        depth_image.save(self.live_depth_path)
        mask_image.save(self.live_mask_path)

        return rgb_image, depth_image, mask_image


    def process_data(self, rgb_image, depth_image, mask_image, camera_prefix):

        if camera_prefix == "d415":
            demo_rgb = np.array(Image.open(self.demo_head_rgb_path))
            demo_depth = np.array(Image.open(self.demo_head_depth_path))
            demo_mask = np.array(Image.open(self.demo_head_mask_path))
        elif camera_prefix == "d405":
            demo_rgb = np.array(Image.open(self.demo_wrist_rgb_path))
            demo_depth = np.array(Image.open(self.demo_wrist_depth_path))
            demo_mask = np.array(Image.open(self.demo_wrist_mask_path))

        live_rgb = np.array(rgb_image)
        live_depth = np.array(depth_image)
        live_mask = np.array(mask_image)
        
        intrinsics = np.load(self.intrinsics_d415_path) if camera_prefix == "d415" else np.load(self.intrinsics_d405_path)

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

    def estimate_pose(self, data, camera_prefix):

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

        # Global registration using FGR
        
            # Estimate normals for each point cloud
            # pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=40))
            # pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=40))


        if camera_prefix == "d405":
            # Compute FPFH features
            voxel_size = 0.001 # Set voxel size for downsampling (adjust based on your data)

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
        
            # Global registration using FGR
            distance_threshold = voxel_size * 0.5
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold))


            # Use the result of global registration as the initial transformation for ICP
            trans_init = result.transformation

        else:
            # Compute FPFH features
            voxel_size = 0.05 # Set voxel size for downsampling (adjust based on your data)
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
        threshold = 0.01  # Set a threshold for ICP, this depends on your data
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Get the transformation matrix
        T_delta_cam = reg_p2p.transformation


        # Draw the result
        # draw_registration_result(pcd0, pcd1, T_delta_cam)
        # rospy.loginfo(f"ICP Fitness: {reg_p2p.fitness}")
        # rospy.loginfo(f"ICP Inlier RMSE: {reg_p2p.inlier_rmse}")
        
        return T_delta_cam

    def run(self, output_path, camera_prefix):
        rgb_image, depth_image, mask_image = self.inference_and_save(camera_prefix, output_path)
        data = self.process_data(rgb_image, depth_image, mask_image, camera_prefix)
        
        if camera_prefix == 'd415':
            return np.mean(data["pc1"][:, :3], axis=0) - np.mean(data["pc0"][:, :3], axis=0) 
        
        return self.estimate_pose(data, camera_prefix)

if __name__ == '__main__':
    rospy.init_node('PoseEstimation', anonymous=True)
    dir = "experiments/lego_split"

    pose_estimator = PoseEstimation(
        dir=dir,
        text_prompt="lego",
    )
    try:
        T_delta_world = pose_estimator.run(output_path=f"{dir}/", camera_prefix="d415")

    except Exception as e:
        print(f"Error: {e}")

