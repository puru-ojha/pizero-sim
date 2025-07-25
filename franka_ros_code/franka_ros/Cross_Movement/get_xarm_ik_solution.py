#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import json
import os
import tf
from tf.transformations import quaternion_from_euler, quaternion_multiply

def get_first_xarm_ik_solution():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('get_xarm_ik', anonymous=True)

    arm_group_name = "xarm7"
    arm_move_group = moveit_commander.MoveGroupCommander(arm_group_name)
    arm_move_group.set_planning_time(10.0) # Give it some time to find IK

    poses_filepath = "/home/gunjan/catkin_ws/src/franka_ros_code/franka_ros/Cross_Movement/Franka_Recording/franka_poses.json"

    try:
        with open(poses_filepath, 'r') as f:
            pose_list_raw = json.load(f)
    except IOError as e:
        rospy.logerr("Failed to read file {}: {}".format(poses_filepath, e))
        return None

    if not pose_list_raw:
        rospy.logerr("franka_poses.json is empty.")
        return None

    # Transformation from Franka base to xArm base (copy from playback_xarm_from_poses.py)
    q_rotation = quaternion_from_euler(0, 0, -1.570796)

    p_raw = pose_list_raw[0] # Get the first Franka pose
    p_franka = p_raw['position']
    o_franka = p_raw['orientation']

    # Apply translation
    p_translated = {
        'x': p_franka['x'],
        'y': p_franka['y'] - 0.2,
        'z': p_franka['z']
    }

    # Apply rotation to the position vector
    p_rotated_x = p_translated['y']
    p_rotated_y = -p_translated['x']

    # Apply rotation to the orientation quaternion
    q_franka = [o_franka['x'], o_franka['y'], o_franka['z'], o_franka['w']]
    q_rotated = quaternion_multiply(q_rotation, q_franka)

    target_pose = Pose()
    target_pose.position.x = p_rotated_x
    target_pose.position.y = p_rotated_y
    target_pose.position.z = p_franka['z'] # Z is unaffected
    target_pose.orientation.x = q_rotated[0]
    target_pose.orientation.y = q_rotated[1]
    target_pose.orientation.z = q_rotated[2]
    target_pose.orientation.w = q_rotated[3]

    rospy.loginfo("Attempting to find IK for target pose:\n{}".format(target_pose))

    # Get current joint values to use as seed for IK
    current_joint_values = arm_move_group.get_current_joint_values()
    rospy.loginfo("Current joint values: {}".format(current_joint_values))

    # Set the target pose and try to get an IK solution
    arm_move_group.set_pose_target(target_pose)
    plan_success, plan, _, _ = arm_move_group.plan()

    if plan_success:
        rospy.loginfo("Found a plan to the target pose. First joint state of the plan:")
        first_joint_state = plan.joint_trajectory.points[0].positions
        rospy.loginfo(first_joint_state)
        return first_joint_state
    else:
        rospy.logerr("Failed to find a plan to the target pose. No IK solution found.")
        return None

if __name__ == '__main__':
    try:
        ik_solution = get_first_xarm_ik_solution()
        if ik_solution:
            rospy.loginfo("Use these joint values in xarm_gazebo/launch/xarm7_new.launch:")
            rospy.loginfo(ik_solution)
        else:
            rospy.logerr("Could not get IK solution.")
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
