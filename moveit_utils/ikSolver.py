import rospy
from moveit_commander import RobotCommander
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

class IKSolver:
    def __init__(self, group_name, ik_link_name):
        self.group_name = group_name
        self.ik_link_name = ik_link_name
        self.robot = RobotCommander()

        rospy.wait_for_service('compute_ik')
        self.compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

        if not rospy.get_node_uri():
            rospy.init_node('ik_solver_node', anonymous=True)
        rospy.loginfo("IK Solver initialized for group: {} and link: {}".format(group_name, ik_link_name))

    def get_ik(self, pose):
        request = GetPositionIKRequest()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.ik_link_name
        request.ik_request.pose_stamped.header.frame_id = "world"
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state = self.robot.get_current_state()

        # Filling the PoseStamped message
        request.ik_request.pose_stamped.pose = pose

        try:
            response = self.compute_ik(request)  # type: GetPositionIKResponse
            # rospy.loginfo("IK successfully computed")
            return response
        except rospy.ServiceException as e:
            rospy.logerr('Service call failed: {}'.format(e))

