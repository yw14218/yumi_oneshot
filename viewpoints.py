import rospy
import yumi_moveit_utils as yumi
from std_srvs.srv import Empty
import numpy as np
from geometry_msgs.msg import Pose, Vector3, TransformStamped
from tf.transformations import quaternion_from_euler
from math import pi, cos , sin
import tf2_ros
from scipy.spatial.transform import Rotation as R

def generate_hemisphere_poses_around_point(point, radius, num_poses):
    """
    Generate a list of hemisphere camera poses around a 3D point.

    Parameters:
    - point: 3D coordinate (x, y, z) on the table
    - radius: Radius of the hemisphere
    - num_poses: Number of camera poses to generate

    Returns:
    - List of camera poses (each pose represented as a tuple of (x, y, z, roll, pitch, yaw))
    """
    x, y, z = point

    # Generate uniformly distributed points on a sphere (hemisphere)
    phi = np.linspace(0, np.pi/2, num_poses)  # inclination (polar angle)
    theta = np.linspace(0, 2*np.pi, num_poses)  # azimuth (azimuthal angle)
    
    # Generate camera poses around the hemisphere
    camera_poses = []
    for incline in phi:
        for azimuth in theta:
            pose_x = x + radius * np.sin(incline) * np.cos(azimuth)
            pose_y = y + radius * np.sin(incline) * np.sin(azimuth)
            pose_z = z + radius * np.cos(incline)

            # Calculate look direction
            look_dir = np.array(point) - np.array([pose_x, pose_y, pose_z])
            look_dir = look_dir / np.linalg.norm(look_dir)

            # Sample an arbitrary, roughly upward direction 
            up_dir = np.array([0, 0, 1]) 

            # Create a quaternion 
            quat = R.from_rotvec(np.arctan2(look_dir[0], look_dir[2]), up_dir).as_quat()

            camera_poses.append((pose_x, pose_y, pose_z, *quat))  # *quat unpacks the quaternion

    return camera_poses

def run():
    """Starts the node

    Runs to start the node and initialize everthing. Runs forever via Spin()

    :returns: Nothing
    :rtype: None
    """


    object_coords = np.array([0.5, 0, 0.15])
    yumi.static_tf_broadcast("world", "object", [0.5, 0, 0.15, 0, 0, 0, 1])
    
    camera_poses = generate_hemisphere_poses_around_point(object_coords, 0.2, 10)
    print(camera_poses)
    tf_set = []
    for i, camera_pose in enumerate(camera_poses):
        static_transformStamped = TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = f"camera_pose_{i}"
        static_transformStamped.transform.translation.x = camera_pose[0]
        static_transformStamped.transform.translation.y = camera_poses[1]
        static_transformStamped.transform.translation.z = camera_poses[2]
        static_transformStamped.transform.rotation.x = camera_poses[3]
        static_transformStamped.transform.rotation.y = camera_poses[4]
        static_transformStamped.transform.rotation.z = camera_poses[5]
        static_transformStamped.transform.rotation.w = camera_poses[6]
        tf_set.append(static_transformStamped)

    br = tf2_ros.StaticTransformBroadcaster()
    br.sendTransform(tf_set)
    rospy.spin()




if __name__ == '__main__':
    rospy.init_node('yumi_moveit_demo')
    try:
       run()
    except Exception as e:
        print(f"Error: {e}")