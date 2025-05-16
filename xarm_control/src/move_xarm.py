#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from actionlib import SimpleActionClient

# Initialize ROS node
rospy.init_node('xarm_joint_commander', anonymous=True)

# Create an action client to send joint commands
action_client = SimpleActionClient('/xarm/xarm7_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
action_client.wait_for_server(rospy.Duration(5)) # Wait for the action server to come up

# Define joint names (xArm 7 has 7 joints)
joint_names = [
    "joint1", "joint2", "joint3", "joint4", 
    "joint5", "joint6", "joint7"
]

# Define your target joint angles (in radians)
target_angles = [0.5, -0.7, 0.8, 0.5, 0.3, 0.9, 0.2] # Example values

# Create JointTrajectory message
traj_msg = JointTrajectory()
traj_msg.joint_names = joint_names

# Create a trajectory point
point = JointTrajectoryPoint()
point.positions = target_angles
point.time_from_start = rospy.Duration(10.0)  # Move in 2 seconds

# Append the point to the trajectory
traj_msg.points.append(point)

# Create a goal
goal_msg = FollowJointTrajectoryGoal()
goal_msg.trajectory = traj_msg

# Publish the command
rospy.sleep(1)  # Give some time for connection
rospy.loginfo("Sending trajectory")
action_client.send_goal(goal_msg)
action_client.wait_for_result()
result = action_client.get_result()

# Check if the goal was successful
if result.error_code == 0:
    rospy.loginfo(f'Goal Succeeded')
else:
    rospy.loginfo('Goal failed')

rospy.sleep(3)  # Wait for execution
