#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, JointState, CameraInfo
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from actionlib import SimpleActionClient
import numpy as np
from cv_bridge import CvBridge
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import time

class CameraJointControlNode:
    def __init__(self):
        rospy.init_node('camera_joint_control_node', anonymous=True)
        rospy.loginfo('camera_joint_control_node started')

        # Initialize the bridge between ROS and OpenCV
        self.bridge = CvBridge()

        # --- Camera Subscriptions ---
        # Gazebo Camera
        self.camera_image_subscriber_gazebo = rospy.Subscriber(
            '/xarm_exterior_camera/image_raw', Image, self.camera_image_callback_gazebo, queue_size=10)
        self.camera_info_subscriber_gazebo = rospy.Subscriber(
            '/xarm_exterior_camera/camera_info', CameraInfo, self.camera_info_callback_gazebo, queue_size=10)

        # RealSense Camera (Color)
        self.camera_image_subscriber_realsense_color = rospy.Subscriber(
            '/realsense_gazebo_camera/color/image_raw', Image, self.camera_image_callback_realsense_color, queue_size=10)
        self.camera_info_subscriber_realsense_color = rospy.Subscriber(
            '/realsense_gazebo_camera/color/camera_info', CameraInfo, self.camera_info_callback_realsense_color, queue_size=10)

        # --- Joint State Subscription ---
        self.joint_state_subscriber = rospy.Subscriber(
            '/xarm/joint_states', JointState, self.joint_state_callback, queue_size=10)

        # --- Action Client (Trajectory Execution) ---
        self.action_client = SimpleActionClient('/xarm/xarm7_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.action_server_connected = False
        if self.action_client.wait_for_server(rospy.Duration(15)):
            rospy.loginfo("Action server connected")
            self.action_server_connected = True
        else:
            rospy.logerr("Action server did not come up, not sending goals.")
        rospy.sleep(2) # wait for 2 more seconds


        # --- Variables to store data ---
        self.joint_positions = None
        self.camera_image_gazebo = None
        self.camera_info_gazebo = None
        self.camera_image_realsense_color = None
        self.camera_info_realsense_color = None
        self.gripper_state = 0 # initialize the gripper state

        self.joint_names = [
        "joint1", "joint2", "joint3", "joint4", 
        "joint5", "joint6", "joint7"
        ]

        # Move the robot to the default position
        if self.action_server_connected: #only move to default position if the connection is ok.
            self.move_to_default_position()
        
        # self.control_loop()

        # --- Websocket Client ---
        self.policy_client = None # initialize the client as None
        try:
            self.policy_client = WebsocketClientPolicy(host="10.4.25.44", port=8000)
            rospy.loginfo("Policy server connected")
        except Exception as e:
            rospy.logerr(f"Could not connect to policy server: {e}")

        # Start control loop
        if self.action_server_connected and self.policy_client is not None:
            print("action server is connected and policy client is also there") #only start the control loop if the connection is ok.
            self.control_loop()

    # --- Camera Callbacks ---
    def camera_image_callback_gazebo(self, msg):
        #rospy.loginfo(f'Received camera image (gazebo) with size: {len(msg.data)}')
        # Convert the image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.camera_image_gazebo = cv_image

    def camera_info_callback_gazebo(self, msg):
        #rospy.loginfo(f'Received camera info (gazebo)')
        self.camera_info_gazebo = msg

    def camera_image_callback_realsense_color(self, msg):
        #rospy.loginfo(f'Received camera image (realsense color) with size: {len(msg.data)}')
        # Convert the image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.camera_image_realsense_color = cv_image

    def camera_info_callback_realsense_color(self, msg):
        #rospy.loginfo(f'Received camera info (realsense color)')
        self.camera_info_realsense_color = msg

    # --- Joint State Callback ---
    def joint_state_callback(self, msg):
        #rospy.loginfo(f'Received joint states:')
        self.joint_positions = np.array(msg.position)


    # --- Action Client Methods ---
    def send_trajectory(self, trajectory):
        # Create a goal
        goal_msg = FollowJointTrajectoryGoal()
        goal_msg.trajectory = trajectory

        # Send the goal
        rospy.loginfo('Sending trajectory')
        if self.action_server_connected: #only send the goal if the connection is ok.
            self.action_client.send_goal(goal_msg)
            self.action_client.wait_for_result() # wait for the result.
            result = self.action_client.get_result()
            if result:
                rospy.loginfo(f"result: {result}")
        else:
            rospy.logerr(f"Action server is down, could not send the goal: {goal_msg}")
    
    def move_to_default_position(self):
        rospy.loginfo("Moving to default position...")
        target_angles = [0.0, -0.7, 0.0, 0.5, 0.0, 0.9, 0.8] # Example values

        # Create JointTrajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        # Create a trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(5.0)  # Move in 5 seconds
        rospy.loginfo(f'Joint trajectory point: {point.positions}') #print the trajectory

        # Append the point to the trajectory
        traj_msg.points.append(point)

        self.send_trajectory(traj_msg)
        rospy.sleep(5) #Wait for the robot to arrive at the position
        rospy.loginfo("Arrived at default position")

    def control_loop(self):
        rospy.loginfo('Starting main control loop')
        rate = rospy.Rate(1)  # 1 Hz - Adjust the frequency as needed
        
        while not rospy.is_shutdown():
            # Prepare observation data for the policy
            observation = {
                "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(self.camera_image_gazebo, 224, 224)
                ),
                "observation/wrist_image_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(self.camera_image_realsense_color, 224, 224)
                ),
                "observation/joint_position": self.joint_positions,
                "observation/gripper_position": [self.gripper_state], #add gripper state
                "prompt": "Move the gripper to the left side of the table",
            }

            # --- Added code to inspect observation dimensions ---
            rospy.loginfo("--- Observation Dictionary Dimensions ---")
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    rospy.loginfo(f"{key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    rospy.loginfo(f"{key}: type=list, length={len(value)}")
                    if len(value) >0 and isinstance(value[0], (int, float)) :
                        rospy.loginfo(f"   -Example element : {value[0]} dtype={type(value[0])}")
                elif isinstance(value, tuple):  # Handle tuples
                    rospy.loginfo(f"{key}: type=tuple, length={len(value)}")
                    if len(value) >0 and isinstance(value[0], (int, float)):
                        rospy.loginfo(f"   -Example element : {value[0]} dtype={type(value[0])}")
                elif isinstance(value, (int,float)):
                    rospy.loginfo(f"{key}: type={type(value)}, value={value}")#print the value.
                else:
                    rospy.loginfo(f"{key}: type={type(value)}")
            rospy.loginfo("---------------------------------------")
            # --- End of added code ---
            if self.policy_client is not None:
                # Get the trajectory from the policy
                start = time.time()
                action_chunk = self.policy_client.infer(observation)["actions"]
                end = time.time()

                rospy.loginfo(f"Inference time: {end-start}")

                # Convert action chunk to JointTrajectory
                trajectory, gripper_state = self.convert_action_to_trajectory(action_chunk)
                self.gripper_state = gripper_state
                # Send the trajectory to the xArm
                self.send_trajectory(trajectory)
            else:
                rospy.logwarn("Policy client is not connected, skipping inference")
            rate.sleep()

    def convert_action_to_trajectory(self, action_chunk):
        """Converts the action chunk received from the server to a JointTrajectory."""
        # Assuming action chunk is a numpy array of 7 joint angles (or a list)
        if isinstance(action_chunk, list):
            action_chunk = np.array(action_chunk)

        # The dimensions of action_chunk is
        print(action_chunk.shape)
        
        target_angles = action_chunk[0][:7]  # use the first 7 value in the array, for the joints
        gripper_state = int(action_chunk[0][7]) # the eigth value will be the gripper position

        # Create JointTrajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        # Create a trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(5)  # Move in 5 seconds

        # Append the point to the trajectory
        traj_msg.points.append(point)

        return traj_msg, gripper_state
if __name__ == '__main__':
    try:
        CameraJointControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
