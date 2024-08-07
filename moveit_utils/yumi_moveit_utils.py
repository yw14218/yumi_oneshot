#!/usr/bin/env python

import sys
import copy
import rospy
import tf
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import tf2_ros
import threading
from yumi_hw.srv import *
from moveit_commander import *
from geometry_msgs.msg import TransformStamped
from trajectory_utils import create_homogeneous_matrix

LEFT = 2        #:ID of the left arm
RIGHT = 1       #:ID of the right arm
BOTH = 3        #:ID of both_arms
PI = 3.1415926  #:Value of PI
TORELANCE = 0.0005

table_height = 0.09 #:The height of the upper surface of the table
table_distance_x = 0.74

global group_l  #:The move group for the left arm
global group_r  #:The move group for the right arm
global group_both #:The move group for using both arms at once
# global robot    #:The RobotCommander() from MoveIt!
global scene    #:The PlanningSceneInterface from MoveIt!
global display_trajectory_publisher

# ============ Left Reference frame: world ============ 
# ============ Right Reference frame: world ============ 
# Current pose reference frame for group_l: world
# ============ Left End effector: gripper_l_base ============ 
# ============ Right End effector: gripper_r_base ============ 


#Initializes the package to interface with MoveIt!
def init_Moveit():
    """Initializes the connection to MoveIt!

    Initializes all the objects related to MoveIt! functions. Also adds in the
    table to the MoveIt! planning scene.

    :returns: Nothing
    :rtype: None
    """

    global group_l
    global group_r
    global group_both
    global robot
    global scene
    print("####################################     Start Initialization     ####################################")
    moveit_commander.roscpp_initialize(sys.argv)
    
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    rospy.sleep(1)

    #clean the scene
    scene.remove_world_object("table")
    scene.remove_world_object("part")

    # publish a demo scene
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()

    # add a table
    p.pose.position.x = table_distance_x
    p.pose.position.y = 0.0
    p.pose.position.z = table_height
    scene.add_box("table", p, (1, 2, 0.1))

    # add an object to be grasped
    # p.pose.position.x = 0.205
    # p.pose.position.y = -0.12
    # p.pose.position.z = 0.7
    # scene.add_box("part", p, (0.07, 0.01, 0.2))

    group_l = moveit_commander.MoveGroupCommander("left_arm")
    group_l.set_planner_id("ESTkConfigDefault")
    group_l.set_pose_reference_frame("world")
    group_l.allow_replanning(False)
    group_l.set_goal_position_tolerance(TORELANCE)
    group_l.set_goal_orientation_tolerance(TORELANCE)
    group_l.set_max_velocity_scaling_factor(0.2)

    group_r = moveit_commander.MoveGroupCommander("right_arm")
    group_r.set_planner_id("ESTkConfigDefault")
    group_r.set_pose_reference_frame("world")
    group_r.allow_replanning(False)
    group_r.set_goal_position_tolerance(TORELANCE)
    group_r.set_goal_orientation_tolerance(TORELANCE)
    group_r.set_max_velocity_scaling_factor(0.2)

    group_both = moveit_commander.MoveGroupCommander("both_arms")
    group_both.set_planner_id("ESTkConfigDefault")
    group_both.set_pose_reference_frame("world")
    group_both.allow_replanning(False)
    group_both.set_goal_position_tolerance(TORELANCE)
    group_both.set_goal_orientation_tolerance(TORELANCE)
    group_both.set_max_velocity_scaling_factor(0.2)


    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', 	moveit_msgs.msg.DisplayTrajectory, queue_size=20)
    rospy.sleep(3)
    print("####################################     Finished Initialization     ####################################")

    sys.stdout.write('\nYuMi MoveIt! demo initialized!\n\n\n')




def print_current_pose(arm):
    """Prints the current pose

    Prints the current pose of the selected arm into the terminal

    :param arm: The arm to display the pose of (RIGHT or LEFT)
    :type arm: int
    :returns: Nothing
    :rtype: None
    """
    pose = get_current_pose(arm)
    rpy = get_current_rpy(arm)

    if (arm == LEFT):
        print("Left arm pose and rpy:")
    if (arm == RIGHT):
        print("Right arm pose and rpy:")

    print(pose)
    print(rpy)




def get_current_pose(arm):
    """Gets the current pose

    Return the current pose of the selected arm

    :param arm: The arm to display the pose of (RIGHT or LEFT)
    :type arm: int
    :returns: Pose
    :rtype: PoseStamped
    """
    global group_l
    global group_r
    if (arm == LEFT):
        cur_arm = group_l
    if (arm == RIGHT):
        cur_arm = group_r
    return cur_arm.get_current_pose()




def get_current_rpy(arm):
    """Gets the current orientation

    Returns the current orientation of the selected arm as Euler angles

    :param arm: The arm to display the pose of (RIGHT or LEFT)
    :type arm: int
    :returns: Orientation
    :rtype: Orientation
    """
    global group_l
    global group_r
    if (arm == LEFT):
        cur_arm = group_l
    if (arm == RIGHT):
        cur_arm = group_r
    return cur_arm.get_current_rpy()




def get_current_joint_values(sel):
    """Gets the current joint values

    Returns the current position of all joints in the selected arm. From J1-J7

    :param sel: The arm to display the pose of (RIGHT or LEFT)
    :type sel: int
    :returns: Joint values
    :rtype: float[]
    """
    global group_l
    global group_r

    if (sel == RIGHT):
        cur_arm = group_r
    if (sel == LEFT):
        cur_arm = group_l

    return cur_arm.get_current_joint_values()




def print_current_joint_states(arm):
    """Prints the current joint values

    Prints the current joint values of the selected arm into the terminal

    :param arm: The arm to display the pose of (RIGHT or LEFT)
    :type arm: int
    :returns: Nothing
    :rtype: None
    """

    if (arm == LEFT):
        print("Left arm joint states:")
    if (arm == RIGHT):
        print("Right arm joint states:")

    print(get_current_joint_values(arm))




#Create a pose object from position and orientation (euler angles)
def create_pose_euler(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad):
    """Creates a pose using euler angles

    Creates a pose for use with MoveIt! using XYZ coordinates and RPY
    orientation in radians

    :param x_p: The X-coordinate for the pose
    :param y_p: The Y-coordinate for the pose
    :param z_p: The Z-coordinate for the pose
    :param roll_rad: The roll angle for the pose
    :param pitch_rad: The pitch angle for the pose
    :param yaw_rad: The yaw angle for the pose
    :type x_p: float
    :type y_p: float
    :type z_p: float
    :type roll_rad: float
    :type pitch_rad: float
    :type yaw_rad: float
    :returns: Pose
    :rtype: PoseStamped
    """
    quaternion = tf.transformations.quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
    return create_pose(x_p, y_p, z_p, quaternion[0], quaternion[1], quaternion[2], quaternion[3])




#Create a pose object from position and orientation (quaternion)
def create_pose(x_p, y_p, z_p, x_o, y_o, z_o, w_o):
    """Creates a pose using quaternions

    Creates a pose for use with MoveIt! using XYZ coordinates and XYZW
    quaternion values

    :param x_p: The X-coordinate for the pose
    :param y_p: The Y-coordinate for the pose
    :param z_p: The Z-coordinate for the pose
    :param x_o: The X-value for the orientation
    :param y_o: The Y-value for the orientation
    :param z_o: The Z-value for the orientation
    :param w_o: The W-value for the orientation
    :type x_p: float
    :type y_p: float
    :type z_p: float
    :type x_o: float
    :type y_o: float
    :type z_o: float
    :type w_o: float
    :returns: Pose
    :rtype: PoseStamped
    """
    pose_target = geometry_msgs.msg.Pose()
    pose_target.position.x = x_p
    pose_target.position.y = y_p
    pose_target.position.z = z_p
    pose_target.orientation.x = x_o
    pose_target.orientation.y = y_o
    pose_target.orientation.z = z_o
    pose_target.orientation.w = w_o
    return pose_target




#Set the gripper to an effort value
def gripper_effort(gripper_id, effort):
    """Set gripper effort

    Sends an effort command to the selected gripper. Should be in the range of
    -20.0 (fully open) to 20.0 (fully closed)

    :param gripper_id: The ID of the selected gripper (LEFT or RIGHT)
    :param effort: The effort value for the gripper (-20.0 to 20.0)
    :type gripper_id: int
    :type effort: float
    :returns: Nothing
    :rtype: None
    """
    
    # rospy.loginfo("Setting gripper " + str(gripper_id) + " to " + str(effort))
    # rospy.loginfo('Setting gripper effort to ' + str(effort) + ' for arm ' + str(gripper_id))
    
    if (gripper_id == RIGHT):
        pubname = '/yumi/gripper_r_effort_cmd'

    if (gripper_id == LEFT):
        pubname = '/yumi/gripper_l_effort_cmd'

    pub = rospy.Publisher(pubname, std_msgs.msg.Float64, queue_size=10, latch=True)
    pub.publish(std_msgs.msg.Float64(effort))
    rospy.sleep(1)




#Wrapper for plan_and_move, just position, orientation and arm
def go_to_simple(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad, arm):
    """Set effector position

    Sends a command to MoveIt! to move to the selected position, in any way
    it sees fit.

    :param x_p: The X-coordinate for the pose
    :param y_p: The Y-coordinate for the pose
    :param z_p: The Z-coordinate for the pose
    :param roll_rad: The roll angle for the pose
    :param pitch_rad: The pitch angle for the pose
    :param yaw_rad: The yaw angle for the pose
    :param arm: The arm to execute the movement
    :type x_p: float
    :type y_p: float
    :type z_p: float
    :type roll_rad: float
    :type pitch_rad: float
    :type yaw_rad: float
    :type arm: int
    :returns: Nothing
    :rtype: None
    """
    if (arm == LEFT):
        plan_and_move(group_l, create_pose_euler(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad))
    elif (arm == RIGHT):
        plan_and_move(group_r, create_pose_euler(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad))




#Make a plan and move within the joint space
def go_to_joints(positions, arm):
    """Set joint values

    Moves the selected arm to make the joint positions match the given values

    :param positions: The desired joint values [j1-j7] (or [j1l-j7l,j1r-j7r] for both arms at once)
    :param arm: The selected arm (LEFT, RIGHT or BOTH)
    :type positions: float[]
    :type arm: int
    :returns: Nothing
    :rtype: None
    """
    global group_l
    global group_r
    global group_both

    cur_arm = group_l
    if (arm == RIGHT):
        cur_arm = group_r
    elif (arm == BOTH):
        cur_arm = group_both

    if (arm == BOTH):
        cur_arm.set_joint_value_target(positions[0] + positions[1])
    else:
        cur_arm.set_joint_value_target(positions)
    cur_arm.plan()
    cur_arm.go(wait=True)
    rospy.sleep(3)




def plan_path(points, arm, planning_tries = 500):
    global robot
    global group_l
    global group_r

    if (arm == LEFT):
        armname = 'left'
        cur_arm = group_l
    if (arm == RIGHT):
        armname = 'right'
        cur_arm = group_r

    rospy.loginfo('Moving ' + armname + ' through path: ')
    rospy.loginfo(points)

    waypoints = []
    #waypoints.append(cur_arm.get_current_pose().pose)
    for point in points:
        if len(point) == 6:
            wpose = create_pose_euler(point[0], point[1], point[2], point[3], point[4], point[5])
        else:
            wpose = point
        waypoints.append(copy.deepcopy(wpose))

    rospy.loginfo(waypoints)

    cur_arm.set_start_state_to_current_state()

    attempts = 0
    fraction = 0.0

    while fraction < 1.0 and attempts < planning_tries:
        (plan, fraction) = cur_arm.compute_cartesian_path(waypoints, 0.01, 0.0, True)
        attempts += 1
        print(attempts)
        rospy.loginfo('attempts: ' + str(attempts) + ', fraction: ' + str(fraction))
        if (fraction == 1.0):
            plan = cur_arm.retime_trajectory(robot.get_current_state(), plan, 1.0)
            return plan
            #r = cur_arm.execute(plan)

    if (fraction < 1.0):
        rospy.logerr('Only managed to calculate ' + str(fraction*100) + '% of the path!')
        raise Exception('Could not calculate full path, exiting')

    return None




#Plan a cartesian path through the given waypoints and execute
def traverse_path(points, arm, planning_tries = 500):
    """Commands an end-effector to traverse a path

    Creates a path between the given waypoints, interpolating points spaced
    by 1cm. Then tries to plan a trajectory through those points, until it
    succeeds and executes or if it fails to find a path for 500 tries it throws
    an exception.

    :param points: An array of waypoints which are themselves array of the form [x,y,z,r,p,y]
    :param arm: The selected arm (LEFT or RIGHT)
    :param planning_tries: The number of tries allowed for planning (default=100)
    :type points: float[[]]
    :type arm: int
    :type planning_tries: int
    :returns: Nothing
    :rtype: None
    """
    global group_l
    global group_r

    if (arm != BOTH):
        plan = plan_path(points, arm, planning_tries)
        if (plan != None):
            if(arm == RIGHT):
                group_r.execute(plan)
            else:
                group_l.execute(plan)
    else:
        # traverse_pathDual(points, planning_tries)
        pass




#Plan a trajectory and execute it
def plan(move_group, target):
    """Plans and moves a group to target

    Creates a plan to move a move_group to the given target. Used by go_to_simple

    :param move_group: The group to move
    :param target: The pose to move to
    :type move_group: MoveGroup
    :type target: PoseStamped
    :returns: Nothing
    :rtype: None
    """
    euler = tf.transformations.euler_from_quaternion((target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w))

    if (move_group == group_l):
        arm = 'left'
    if (move_group == group_r):
        arm = 'right'

    rospy.loginfo('Planning and moving ' + arm + ' arm: Position: {' + str(target.position.x) + ';' + str(target.position.y) + ';' + str(target.position.z) +
                                        '}. Rotation: {' + str(euler[0]) + ';' + str(euler[1]) + ';' + str(euler[2]) + '}.')
    move_group.set_pose_target(target)
    plan = move_group.plan()
    return plan




#Plan a trajectory and execute it
def plan_and_move(move_group, target):
    """Plans and moves a group to target

    Creates a plan to move a move_group to the given target. Used by go_to_simple

    :param move_group: The group to move
    :param target: The pose to move to
    :type move_group: MoveGroup
    :type target: PoseStamped
    :returns: Nothing
    :rtype: None
    """

    euler = tf.transformations.euler_from_quaternion((target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w))

    if (move_group == group_l):
        arm = 'left'
    if (move_group == group_r):
        arm = 'right'

    rospy.loginfo('Planning and moving ' + arm + ' arm: Position: {' + str(target.position.x) + ';' + str(target.position.y) + ';' + str(target.position.z) +
                                        '}. Rotation: {' + str(euler[0]) + ';' + str(euler[1]) + ';' + str(euler[2]) + '}.')
    move_group.set_pose_target(target)
    plan = move_group.plan()
    move_group.go(wait=True)
    rospy.sleep(3)




#Resets a single arm to the reset pose
def reset_arm(arm):
    """Resets an arm

    Resets a single arm to its reset position

    :param arm: The selected arm (LEFT or RIGHT)
    :type arm: int
    :returns: Nothing
    :rtype: None
    """
    safeJointPositionR = [1.6379728317260742, 0.20191457867622375, -2.5927578258514404, 0.538416862487793, 2.7445449829101562, 1.5043296813964844, 1.7523150444030762]
    safeJointPositionL = [-1.46564781665802, 0.3302380442619324, 2.507143497467041, 0.7764986753463745, -2.852548837661743, 1.659092664718628, 1.378138542175293]
    global group_l
    global group_r
    global group_both

    if (arm == RIGHT):
        group_r.set_joint_value_target(safeJointPositionR)
        group_r.plan()
        group_r.go(wait=True)
        gripper_effort(RIGHT, -15.0)
        gripper_effort(RIGHT, 0.0)
    elif (arm == LEFT):
        group_l.set_joint_value_target(safeJointPositionL)
        group_l.plan()
        group_l.go(wait=True)
        gripper_effort(LEFT, -15.0)
        gripper_effort(LEFT, 0.0)
    elif (arm == BOTH):
        group_both.set_joint_value_target(safeJointPositionL + safeJointPositionR)
        group_both.go(wait=True)
        gripper_effort(LEFT, -15.0)
        gripper_effort(LEFT, 0.0)
        gripper_effort(RIGHT, -15.0)
        gripper_effort(RIGHT, 0.0)

    rospy.sleep(1)




#Resets the YuMi to a predetermined position
def reset_pose():
    """Resets YuMi

    Moves YuMi arms to an initial joint position with grippers open

    :returns: Nothing
    :rtype: None
    """

    print("Resetting YuMi arms to an initial joint position with grippers open")

    reset_arm(BOTH)


def close_grippers(arm):
    """Closes the grippers.

    Closes the grippers with an effort of 15 and then relaxes the effort to 0.

    :param arm: The side to be closed (moveit_utils LEFT or RIGHT)
    :type arm: int
    :returns: Nothing
    :rtype: None
    """
    gripper_effort(arm, 15.0)
    gripper_effort(arm, 0.0)

def open_grippers(arm):
    """Opens the grippers.

    Opens the grippers with an effort of -15 and then relaxes the effort to 0.

    :param arm: The side to be opened (moveit_utils LEFT or RIGHT)
    :type arm: int
    :returns: Nothing
    :rtype: None
    """
    gripper_effort(arm, -15.0)
    gripper_effort(arm, 0.0)


# Resets both arms to calib
def reset_calib():
    
    safeJointPositionR = [4.137169526075013e-05, -2.268953800201416, -2.356186628341675, 0.5236199498176575, -6.807099998695776e-05, 0.6980842351913452, -4.788856676896103e-05]
    safeJointPositionL = [8.723969949642196e-05, -2.268956184387207, 2.35611629486084, 0.5236671566963196, 4.085124237462878e-05, 0.6981115341186523, 2.6177165636909194e-05]

    global group_both


    group_both.set_joint_value_target(safeJointPositionL + safeJointPositionR)
    group_both.go(wait=True)
    gripper_effort(LEFT, -15.0)
    gripper_effort(LEFT, 0.0)
    gripper_effort(RIGHT, -15.0)
    gripper_effort(RIGHT, 0.0)

    rospy.sleep(0.1)



# Resets both arms to init
def reset_init(arm=None):

    safeJointPositionL = [-0.1346624195575714, -1.2939683198928833, 1.152125597000122, 0.8828295469284058, -1.433053731918335, -0.6675148606300354, -0.33587437868118286]
    safeJointPositionR = [0.13037508726119995, -1.293479084968567, -1.1520023345947266, 0.882483184337616, 1.4386858940124512, -0.6661921739578247, 0.3313935697078705]

    if arm is None:
        group_both.set_joint_value_target(safeJointPositionL + safeJointPositionR)
        group_both.go(wait=True)
        # gripper_effort(LEFT, -15.0)
        # gripper_effort(LEFT, 0.0)
        gripper_effort(RIGHT, -15.0)
        gripper_effort(RIGHT, 0.0)
    elif arm == LEFT:
        group_l.set_joint_value_target(safeJointPositionL)
        group_l.go(wait=True)
        gripper_effort(LEFT, -15.0)
        gripper_effort(LEFT, 0.0)
    elif arm == RIGHT:
        group_r.set_joint_value_target(safeJointPositionR)
        group_r.go(wait=True)
        gripper_effort(RIGHT, -15.0)
        gripper_effort(RIGHT, 0.0)
    else:
        raise NotImplementedError

    rospy.sleep(0.1) 
# Resets both arms to init
def reset_init(arm=None):

    safeJointPositionL = [-0.1346624195575714, -1.2939683198928833, 1.152125597000122, 0.8828295469284058, -1.433053731918335, -0.6675148606300354, -0.33587437868118286]
    safeJointPositionR = [0.13037508726119995, -1.293479084968567, -1.1520023345947266, 0.882483184337616, 1.4386858940124512, -0.6661921739578247, 0.3313935697078705]

    if arm is None:
        group_both.set_joint_value_target(safeJointPositionL + safeJointPositionR)
        group_both.go(wait=True)
        gripper_effort(LEFT, -15.0)
        gripper_effort(LEFT, 0.0)
        gripper_effort(RIGHT, -15.0)
        gripper_effort(RIGHT, 0.0)
    elif arm == LEFT:
        group_l.set_joint_value_target(safeJointPositionL)
        group_l.go(wait=True)
        gripper_effort(LEFT, -15.0)
        gripper_effort(LEFT, 0.0)
    elif arm == RIGHT:
        group_r.set_joint_value_target(safeJointPositionR)
        group_r.go(wait=True)
        gripper_effort(RIGHT, -15.0)
        gripper_effort(RIGHT, 0.0)
    else:
        raise NotImplementedError

    rospy.sleep(0.1) 

def static_tf_broadcast(parent_id, child_id, pose_in_list) -> None:
    br = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = parent_id
    static_transformStamped.child_frame_id = child_id
    static_transformStamped.transform.translation.x = pose_in_list[0]
    static_transformStamped.transform.translation.y = pose_in_list[1]
    static_transformStamped.transform.translation.z = pose_in_list[2]
    static_transformStamped.transform.rotation.x = pose_in_list[3]
    static_transformStamped.transform.rotation.y = pose_in_list[4]
    static_transformStamped.transform.rotation.z = pose_in_list[5]
    static_transformStamped.transform.rotation.w = pose_in_list[6]
    br.sendTransform(static_transformStamped)
    print("TF of target link successfully sent")

def plan_both_arms(left_goal, right_goal):
    group_l.set_pose_target(left_goal)
    group_r.set_pose_target(right_goal)
    plan_left = group_l.plan()
    plan_right = group_r.plan()
    pos_left = [point.positions for point in plan_left[1].joint_trajectory.points]
    pos_right = [point.positions for point in plan_right[1].joint_trajectory.points]
    group_both.set_joint_value_target(pos_left[-1] + pos_right[-1])
    group_both.go(wait=True)

    group_both.stop()
    group_l.clear_pose_targets()
    group_r.clear_pose_targets()

def plan_left_arm(goal):
    group_l.set_pose_target(goal)
    plan = group_l.plan()
    group_l.go(wait=True)

def plan_right_arm(goal):
    group_r.set_pose_target(goal)
    plan = group_r.plan()
    group_r.go(wait=True)

def get_curent_T_left():
    cur_pos_left = get_current_pose(LEFT)
    xyz = [cur_pos_left.pose.position.x, cur_pos_left.pose.position.y, cur_pos_left.pose.position.z]
    quaternion = [cur_pos_left.pose.orientation.x, cur_pos_left.pose.orientation.y, cur_pos_left.pose.orientation.z, cur_pos_left.pose.orientation.w]
    T_eef_world = create_homogeneous_matrix(xyz, quaternion)

    return T_eef_world

def operate_gripper_in_threads(arms, close):
    """
    Close operations on arms in separate threads.
    """
    def grip(arm):
        gripper_effort(arm, 20.0 if close else -20.0)
        
    threads = []
    for arm in arms:
        thread = threading.Thread(target=grip, args=(arm,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

def close_left_open_right_in_threads(arms):
    def grip(arm):
        if arm == LEFT:
            gripper_effort(arm, 20.0)
        elif arm == RIGHT:
            gripper_effort(arm, -20.0)
        
    threads = []
    for arm in arms:
        thread = threading.Thread(target=grip, args=(arm,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()