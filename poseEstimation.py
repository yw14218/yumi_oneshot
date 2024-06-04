import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import copy
from sensor_msgs.msg import Image as ImageMsg
from PoseEst.direct.preprocessor import Preprocessor, pose_inv, SceneData
from langSAM import LangSAMProcessor
from trajectory_utils import create_homogeneous_matrix, euler_from_matrix

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

        self.intrinsics_d415_path = "handeye/intrinsics_d415.npy"
        self.T_WC_path = "handeye/T_WC_head.npy"

        self.intrinsics_d405_path = "handeye/intrinsics_d405.npy"
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


    def process_data(self, rgb_image, depth_image, mask_image, camera_prefix, T_WC=np.eye(4)):

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
            T_WC=T_WC
        )

        processor = Preprocessor()
        data.update(processor(data))
        return data

    def estimate_pose(self, data, camera_prefix):

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
        
        return self.estimate_pose(data, camera_prefix)

    def decouple_run(self, output_path, camera_prefix):
        rgb_image, depth_image, mask_image = self.inference_and_save(camera_prefix, output_path)
        data = self.process_data(rgb_image, depth_image, mask_image, camera_prefix, T_WC = np.load(self.T_WC_path))

        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(data["pc0"][:, :3])
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(data["pc1"][:, :3])
        pcd0_centre = np.mean(data["pc0"][:, :3], axis=0)  
        pcd1_centre = np.mean(data["pc1"][:, :3], axis=0)

        # Compute the difference between the centroids
        diff_xyz = pcd1_centre - pcd0_centre

        # Compute FPFH features
        voxel_size = 0.01 # Set voxel size for downsampling (adjust based on your data)

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
        
        threshold = 0.01  # Set a threshold for ICP, this depends on your data
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Get the transformation matrix
        T_delta_world = reg_p2p.transformation      
        diff_rpy = euler_from_matrix(T_delta_world.copy()).copy()

        return diff_xyz, diff_rpy
    
    def run_image_match(self, output_path, camera_prefix):
        import torch
        import cv2
        import poselib
        from matplotlib import pyplot as plt

        def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
            # Calculate the Homography matrix
            H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
            mask = mask.flatten()

            # Get corners of the first image (image1)
            h, w = img1.shape[:2]
            corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

            # Warp corners to the second image (image2) space
            warped_corners = cv2.perspectiveTransform(corners_img1, H)

            # Draw the warped corners in image2
            img2_with_corners = img2.copy()
            for i in range(len(warped_corners)):
                start_point = tuple(warped_corners[i-1][0].astype(int))
                end_point = tuple(warped_corners[i][0].astype(int))
                cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

            # Prepare keypoints and matches for drawMatches function
            keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
            keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
            matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

            # Draw inlier matches
            img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                        matchColor=(0, 255, 0), flags=2)

            return img_matches
        
        rgb_image, depth_image, mask_image = self.inference_and_save(camera_prefix, output_path)
        live_rgb_seg = np.array(rgb_image) * np.array(mask_image).astype(bool)[..., None]
        if camera_prefix == "d415":
            demo_rgb_seg = np.array(Image.open(f"{self.dir}/demo_head_rgb_seg.png"))
        else:
            demo_rgb_seg = np.array(Image.open(f"{self.dir}/demo_wrist_rgb_seg.png"))
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)
        mkpts_0, mkpts_1 = xfeat.match_xfeat_star(demo_rgb_seg, live_rgb_seg, top_k = 4096)
        canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, demo_rgb_seg, live_rgb_seg)
        plt.figure(figsize=(12,12))
        plt.imshow(canvas[..., ::-1]), plt.show()

        K = np.load(self.intrinsics_d415_path)
        T_WC = np.load(self.T_WC_path)
        def image_to_world(points, K, T_WC, depth=1.0):
            K_inv = np.linalg.inv(K)
            points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
            points_cam = (K_inv @ points_hom.T).T * depth
            points_world = (T_WC @ np.hstack((points_cam, np.ones((points_cam.shape[0], 1)))).T).T
            return points_world[:, :3]

        points1_world = image_to_world(mkpts_0, K, T_WC)
        points2_world = image_to_world(mkpts_1, K, T_WC)

        M, inliers = cv2.estimateAffinePartial2D(points1_world[:, :2], points2_world[:, :2])

        # Construct the 3D transformation matrix
        T_delta_world = np.eye(4)
        T_delta_world[0:2, 0:2] = M[0:2, 0:2]
        T_delta_world[0:2, 3] = M[0:2, 2]

        return T_delta_world

if __name__ == '__main__':
    rospy.init_node('PoseEstimation', anonymous=True)
    dir = "experiments/scissor"

    pose_estimator = PoseEstimation(
        dir=dir,
        text_prompt="black scissor",
        visualize=False
    )
    try:
        T_delta_world = pose_estimator.run(output_path=f"{dir}/", camera_prefix="d415")

    except Exception as e:
        print(f"Error: {e}")

