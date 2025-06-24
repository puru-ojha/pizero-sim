#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, JointState, CameraInfo
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance
from xarm_gripper.msg import MoveAction, MoveGoal
from actionlib import SimpleActionClient # Import for gripper action
import numpy as np
from cv_bridge import CvBridge
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import time

class CameraJointControlNode:
    def __init__(self):



        # Define the joint names for the xArm
        self.joint_names = [
        "joint1", "joint2", "joint3", "joint4", 
        "joint5", "joint6", "joint7"
        ]

        self.open_loop_horizon = 8

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
        self.action_client = SimpleActionClient(
            '/xarm/xarm7_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.action_server_connected = False
        if self.action_client.wait_for_server(rospy.Duration(15)):
            rospy.loginfo("xArm Joint Trajectory Action server connected")
            self.action_server_connected = True
        else:
            rospy.logerr("xArm Joint Trajectory Action server did not come up, not sending goals.")
        rospy.sleep(2) # wait for 2 more seconds

        # --- Gripper Action Client ---
        # Assuming xarm_gripper.msg.MoveAction is available (you might need to add xarm_gripper to your package dependencies)
        
        self.gripper_action_client = SimpleActionClient('/xarm/gripper_move', MoveAction)
        self.gripper_action_client.wait_for_server(rospy.Duration(5)) # Wait for gripper action server


        # --- Variables to store data ---
        self.joint_array = None
        self.joint_positions = None
        self.camera_image_gazebo = None
        self.camera_info_gazebo = None
        self.camera_image_realsense_color = None
        self.camera_info_realsense_color = None
  
        # Gripper is controlled in pulses (0-850). 0=closed, 850=open.
        # We store the last commanded pulse value.
        self.gripper_pulse_state = 0.0

        

        # Move the robot to the default position
        if self.action_server_connected: #only move to default position if the connection is ok.
            self.move_to_default_position()

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
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.camera_image_gazebo = cv_image

    def camera_info_callback_gazebo(self, msg):
        #rospy.loginfo(f'Received camera info (gazebo)')
        self.camera_info_gazebo = msg

    def camera_image_callback_realsense_color(self, msg):
        #rospy.loginfo(f'Received camera image (realsense color) with size: {len(msg.data)}')
        # Convert the image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.camera_image_realsense_color = cv_image

    def camera_info_callback_realsense_color(self, msg):
        #rospy.loginfo(f'Received camera info (realsense color)')
        self.camera_info_realsense_color = msg

    # --- Joint State Callback ---
    def joint_state_callback(self, msg):
        # rospy.loginfo(f"Received joint names: {msg.name}")
        self.joint_array = np.array(msg.position)
        # Make sure to map joints in the correct order
        self.joint_positions = np.array([
            msg.position[msg.name.index(joint)] for joint in self.joint_names
        ])


    # --- Action Client Methods ---
    def send_trajectory(self, trajectory):
        # Create a goal
        goal_msg = FollowJointTrajectoryGoal()
        goal_msg.trajectory = trajectory

        # Send the goal and wait for it to complete (blocking call).
        if self.action_server_connected:
            self.action_client.send_goal(goal_msg)
            # This makes the call blocking, ensuring one action step completes before the next begins.
        #     self.action_client.wait_for_result()
        #     result = self.action_client.get_result()
        #     if result and result.error_code == result.SUCCESSFUL:
        #         rospy.loginfo("Trajectory execution successful.")
        #     else:
        #         rospy.logwarn(f"Trajectory execution failed with error code: {result.error_code if result else 'N/A'}")
        # else:
        #     rospy.logerr(f"Action server is down, could not send goal: {goal_msg}")
    
    def move_to_default_position(self):
        rospy.loginfo("Moving to default position...")
        target_angles = [-2.5, -0.6, 0.5, 1.2, 0.0, 1.2, 0.0] # Initial positions from launch file

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

    def send_xarm_gripper_command(self, target_pulse, pulse_speed=1500):
        """
        Sends a command to the xArm gripper action server.
        target_pulse: int, desired gripper position (0-850, 0=closed, 850=open)
        pulse_speed: int, desired gripper speed (1-5000)
        """
        if self.gripper_action_client.wait_for_server(rospy.Duration(1)): # Short wait for quick checks
            goal = MoveGoal()
            goal.target_pulse = float(target_pulse)
            goal.pulse_speed = float(pulse_speed)
            rospy.loginfo(f"Sending xArm gripper command: position={target_pulse}, speed={pulse_speed}")
            self.gripper_action_client.send_goal(goal)
        else:
            rospy.logwarn("xArm Gripper action server not available, skipping gripper command.")

    def control_loop(self):
        rospy.loginfo('Starting main control loop')
        rate = rospy.Rate(15)  # 15 Hz, should match dt in conversion function
        action_chunks = None

        while not rospy.is_shutdown():
            # Wait for all sensor data to be available
            if self.camera_image_gazebo is None or self.camera_image_realsense_color is None or self.joint_positions is None:
                rospy.logwarn_throttle(5, "Waiting for sensor data...")
                rate.sleep()
                continue

            # Prepare observation data for the policy
            observation = {
                "observation/exterior_image_1_left": 
                    image_tools.resize_with_pad(self.camera_image_gazebo, 224, 224)
                ,
                "observation/wrist_image_left": 
                    image_tools.resize_with_pad(self.camera_image_realsense_color, 224, 224)
                ,
                "observation/joint_position": self.joint_positions,
                # Normalize gripper state for the policy (0-850 -> 0-1)
                "observation/gripper_position": [self.gripper_pulse_state],
                "prompt": "Pick up the red marker and put it in the white bowl",
            }

            # Get new actions from policy if we are done with the last chunk
            if self.policy_client is not None and action_chunks is None:
                start_time = time.time()
                try:
                    action_chunks = self.policy_client.infer(observation)["actions"]
                    end_time = time.time()
                    rospy.loginfo(f"Inference time: {end_time - start_time:.3f}s. Received action chunks with shape: {np.array(action_chunks).shape}")
                except Exception as e:
                    rospy.logerr(f"Failed to get inference from policy server: {e}")
                    action_chunks = None  # Ensure we don't proceed with old data
                    rate.sleep()
                    continue

            # Execute the chunk of actions
            if action_chunks is not None:
                num_actions_to_execute = min(self.open_loop_horizon, len(action_chunks))
                rospy.loginfo(f"Executing {num_actions_to_execute} actions from the chunk.")

                for i in range(num_actions_to_execute):
                    action_step = action_chunks[i]
                    
                    # Convert action step to trajectory and gripper command
                    trajectory, gripper_command = self.convert_action_to_trajectory_and_gripper(action_step)
                    print("the initial position is ", self.joint_positions)
                    print("the action step is", action_step)
                    print("the final position is", trajectory.points)

                    if trajectory is None:
                        rospy.logerr("Failed to convert action to trajectory. Skipping step.")
                        continue
                    
                    # Send commands to the robot
                    self.send_trajectory(trajectory)
                    self.send_xarm_gripper_command(gripper_command)
                    

                    # Maintain control rate
                    rate.sleep()
                
                # We've used this chunk, so clear it to fetch a new one
                action_chunks = None
            else:
                rospy.logwarn_throttle(5, "No valid action chunk available, skipping execution cycle.")
                rate.sleep()

    def convert_action_to_trajectory_and_gripper(self, action_step):
        if isinstance(action_step, list):
            action_step = np.array(action_step)
        
        dt = 1 / 15.0  # Time step in seconds, should match control loop rate

        if self.joint_positions is None:
            rospy.logerr("Current joint positions not available. Cannot create trajectory.")
            return None, None

        current_angles = self.joint_positions
        scale_factor = 1.0  # Adjust scaling as needed

        # Integrate velocities to compute target angles.
        target_angles = current_angles + scale_factor * action_step[:7] * dt
        
        # Check joint limits and clip if necessary
        # target_angles = self.check_joint_limits(target_angles)

        # Process gripper command (last value of action)
        # Policy output for gripper is assumed to be normalized in [0, 1].
        # We map this to the xArm gripper's pulse range [0, 850].
        gripper_val = action_step[7]
        
        # Clip the value to be safe and map to the physical range
        gripper_command = np.clip(gripper_val, 0.0, 1.0) * 850.0

        # Create JointTrajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        # Create a trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(dt)  # Move in dt seconds

        traj_msg.points.append(point)

        return traj_msg, gripper_command

if __name__ == '__main__':
    try:
        CameraJointControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
