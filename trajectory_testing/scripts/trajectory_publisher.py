#!/usr/bin/env python

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def publish_trajectory():
    # Initialize the ROS node
    rospy.init_node('trajectory_publisher', anonymous=True)

    # Create a publisher for the trajectory topic
    pub = rospy.Publisher('/arm_trajectory', JointTrajectory, queue_size=10)

    # Set the publishing rate
    rate = rospy.Rate(10)

    # Define the trajectory message
    trajectory = JointTrajectory()
    trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    # Add some example trajectory points
    for i in range(10):
        point = JointTrajectoryPoint()
        point.positions = [0.1 * i] * 7  # Example: all joints move by 0.1 * i radians
        point.time_from_start = rospy.Duration(i * 0.5)  # 0.5 seconds between points
        trajectory.points.append(point)
    # Publish the trajectory periodically
    while not rospy.is_shutdown():
        pub.publish(trajectory)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_trajectory()
    except rospy.ROSInterruptException:
        pass


