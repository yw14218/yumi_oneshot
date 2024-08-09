import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from PoseEst.direct.preprocessor import Preprocessor, SceneData
from langSAM import LangSAMProcessor
from trajectory_utils import euler_from_matrix
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor, as_completed
from pc_valid import robust_alignment_check

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
        
            # # Global registration using RANSAC
            # distance_threshold = voxel_size * 1.5
            # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            #     source_down, target_down, source_fpfh, target_fpfh, mutual_filter=False,
            #     max_correspondence_distance=distance_threshold,
            #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(), 
            #     ransac_n=4,
            #     checkers=[
            #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), 
            #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            #     ],
            #     criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
            # )

            # # Use the result of global registration as the initial transformation for ICP
            # trans_init = result.transformation
            # Global registration using FGR
            distance_threshold = voxel_size * 1.5
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold))


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
        ProbPPR.draw_registration_result(pcd0, pcd1, T_delta_cam)
        rospy.loginfo(f"ICP Fitness: {reg_p2p.fitness}")
        rospy.loginfo(f"ICP Inlier RMSE: {reg_p2p.inlier_rmse}")
        
        return T_delta_cam

    def run(self, output_path, camera_prefix, probICP = False):
        rgb_image, depth_image, mask_image = self.inference_and_save(camera_prefix, output_path)
        data = self.process_data(rgb_image, depth_image, mask_image, camera_prefix)
        
        if probICP:
            probPPR = ProbPPR(voxel_size=0.005)
            return probPPR.main(data, visualize=True)
        else:
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

class ProbPPR:
    def __init__(self, voxel_size=0.005):
        self.voxel_size = voxel_size
    
    @staticmethod
    def draw_registration_result(source, target, transformation):
        source_temp = source.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target])

    def preprocess_point_cloud(self, pcd):
        try:
            pcd_down = pcd.voxel_down_sample(self.voxel_size)
            pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
            return pcd_down
        except Exception as e:
            print(f"Error in preprocess_point_cloud: {e}")
            return None

    def compute_fpfh_features(self, pcd):
        try:
            radius_feature = self.voxel_size * 5
            return o3d.pipelines.registration.compute_fpfh_feature(
                pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        except Exception as e:
            print(f"Error in compute_fpfh_features: {e}")
            return None

    def execute_fast_global_registration(self, source, target, source_fpfh, target_fpfh):
        try:
            distance_threshold = self.voxel_size * 0.5
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold))
            return result
        except Exception as e:
            print(f"Error in execute_fast_global_registration: {e}")
            return None

    def extract_translation_rotation(self, transformation):
        translation = transformation[:3, 3]
        rotation = transformation[:3, :3]
        return translation, rotation

    def transformations_to_vector(self, transformations):
        vectors = []
        for transformation in transformations:
            translation, rotation = self.extract_translation_rotation(transformation)
            rotation_euler = R.from_matrix(rotation).as_euler('xyz')
            vector = np.concatenate((translation, rotation_euler))
            vectors.append(vector)
        return np.array(vectors)

    def compute_mean_transformation(self, transformations):
        vectors = self.transformations_to_vector(transformations)
        
        # Compute mean translation
        mean_translation = np.mean(vectors[:, :3], axis=0)
        
        # Compute mean rotation using quaternion averaging
        mean_rotation_euler = np.mean(vectors[:, 3:], axis=0)
        mean_rotation_euler /= np.linalg.norm(mean_rotation_euler)
        mean_rotation = R.from_euler('xyz', mean_rotation_euler).as_matrix()
        
        mean_transformation = np.eye(4)
        mean_transformation[:3, :3] = mean_rotation
        mean_transformation[:3, 3] = mean_translation
        
        return mean_transformation

    def compute_covariance_matrix(self, transformations):
        vectors = self.transformations_to_vector(transformations)
        mean_vector = np.mean(vectors, axis=0)
        centered_vectors = vectors - mean_vector
        covariance_matrix = np.cov(centered_vectors, rowvar=False)
        return covariance_matrix

    @staticmethod
    def process_registration(source, target, transformation):
        source_sample = source.random_down_sample(0.8)
        target_sample = target.random_down_sample(0.8)
        
        result = o3d.pipelines.registration.registration_icp(
            source_sample, target_sample, 0.01, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        return result.transformation

    def estimate_registration_uncertainty(self, source, target, transformation, num_iterations=5000):
        try:
            transformations = []
            
            with ThreadPoolExecutor() as executor:
                # Submit all tasks to the executor
                futures = [executor.submit(ProbPPR.process_registration, source, target, transformation) for _ in range(num_iterations)]
                
                for future in as_completed(futures):
                    transformations.append(future.result())
            
            # Check the type and shape of the transformations
            if not transformations:
                raise ValueError("No transformations were collected.")
            
            # Convert list of transformations to a numpy array
            transformations = np.array(transformations)
            
            # Compute mean and covariance matrix
            mean_transformation = self.compute_mean_transformation(transformations)
            covariance_matrix = self.compute_covariance_matrix(transformations)
            
            return mean_transformation, covariance_matrix
            
        except Exception as e:
            print(f"Error in estimate_registration_uncertainty: {e}")
            return None, None

    def main(self, data, confidence_threshold=0.7, visualize=True):
        try:
            pcd0 = o3d.geometry.PointCloud()
            pcd1 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(data["pc0"][:, :3])
            pcd1.points = o3d.utility.Vector3dVector(data["pc1"][:, :3])

            # Define grid search parameters
            voxel_sizes = [0.001, 0.01, 0.1]  # 0.1 cm, 1 cm, 10 cm
            correspondence_distances = [0.01, 0.05, 0.1]  # 1 cm, 5 cm, 10 cm

            best_fitness = -np.inf
            best_rmse = np.inf
            best_transformation = None

            MIN_POINTS = 10
            for voxel_size in voxel_sizes:
                self.voxel_size = voxel_size
                source_down = self.preprocess_point_cloud(pcd0)
                target_down = self.preprocess_point_cloud(pcd1)

                if source_down is None or target_down is None:
                    raise ValueError("Preprocessing failed")

                source_fpfh = self.compute_fpfh_features(source_down)
                target_fpfh = self.compute_fpfh_features(target_down)

                if source_fpfh is None or target_fpfh is None:
                    raise ValueError("Feature computation failed")

                if source_fpfh is None or target_fpfh is None or source_fpfh.data.shape[1] == 0 or target_fpfh.data.shape[1] == 0:
                    print(f"Skipping voxel_size {voxel_size}: Computed FPFH features are empty.")
                    continue

                if source_fpfh.data.shape[1] < MIN_POINTS or target_fpfh.data.shape[1] < MIN_POINTS:
                    print(f"Skipping voxel_size {voxel_size}: Not enough features after computation.")
                    continue

                # Perform Fast Global Registration (FGR)
                try:
                    result_fgr = self.execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh)
                except Exception as e:
                    print(f"Error in execute_fast_global_registration: {e}")
                    continue
                
                if result_fgr is None:
                    print("An error occurred in main: 'NoneType' object has no attribute 'fitness'")
                    continue

                # Record FGR results
                current_fitness = result_fgr.fitness
                current_rmse = result_fgr.inlier_rmse
                current_transformation = result_fgr.transformation

                # Compare with the best found so far
                if current_fitness > best_fitness or (current_fitness == best_fitness and current_rmse < best_rmse):
                    best_fitness = current_fitness
                    best_rmse = current_rmse
                    best_transformation = current_transformation

                # Refine the transformation using ICP
                for max_corr_dist in correspondence_distances:
                    result_icp = o3d.pipelines.registration.registration_icp(
                        source_down, target_down, max_corr_dist, current_transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                    
                    if result_icp is None:
                        raise ValueError("ICP failed")

                    # Evaluate ICP results
                    current_fitness = result_icp.fitness
                    current_rmse = result_icp.inlier_rmse
                    current_transformation = result_icp.transformation

                    # Check if this result is better based on both fitness and RMSE
                    if current_fitness > best_fitness or (current_fitness == best_fitness and current_rmse < best_rmse):
                        best_fitness = current_fitness
                        best_rmse = current_rmse
                        best_transformation = current_transformation

            # source_pcd_np = np.asarray(pcd0.points)
            # target_pcd_np = np.asarray(pcd1.points)

            # # Convert the NumPy arrays to Open3D tensors
            # source_pcd_tensor = o3d.core.Tensor(source_pcd_np, dtype=o3d.core.float32)
            # target_pcd_tensor = o3d.core.Tensor(target_pcd_np, dtype=o3d.core.float32)

            # # Initialize PointClouds with the tensor positions
            # source_pcd = o3d.t.geometry.PointCloud(source_pcd_tensor)
            # target_pcd = o3d.t.geometry.PointCloud(target_pcd_tensor)

            # information_matrix = o3d.t.pipelines.registration.get_information_matrix(source_pcd, target_pcd, 0.005, result_icp.transformation)
            # # print(information_matrix)

            mean_transformation, covariance_matrix = self.estimate_registration_uncertainty(
                source_down, target_down, best_transformation)

            # source = o3d.geometry.PointCloud()
            # target = o3d.geometry.PointCloud()
            # source.points = o3d.utility.Vector3dVector(data["pc0"][:, :3])
            # target.points = o3d.utility.Vector3dVector(data["pc1"][:, :3])
            # # Perform robust alignment check
            # is_off, reason, metrics = robust_alignment_check(source, target, best_transformation, best_fitness)
            
            # print("Alignment Metrics:")
            # for key, value in metrics.items():
            #     print(f"{key}: {value:.4f}")
            
            # print(f"\nIs alignment extremely off? {'Yes' if is_off else 'No'}")
            # print(f"Reason: {reason}")

            if mean_transformation is None or covariance_matrix is None:
                raise ValueError("Uncertainty estimation failed")

            print(covariance_matrix)
            if visualize:
                self.draw_registration_result(pcd0, pcd1, best_transformation)

            return best_transformation, covariance_matrix

        except Exception as e:
            print(f"An error occurred in main: {e}")
            return None, None
        
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

