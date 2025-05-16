cmake_minimum_required(VERSION 3.0.2)
project(xarm_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  trajectory_msgs
  control_msgs
  actionlib
  cv_bridge
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES xarm_control
#  CATKIN_DEPENDS rospy sensor_msgs trajectory_msgs control_msgs actionlib cv_bridge
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#install the scripts
install(PROGRAMS
  src/camera_joint_control_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
