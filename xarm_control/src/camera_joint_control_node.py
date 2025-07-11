#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, JointState, CameraInfo
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from actionlib import SimpleActionClient
from scipy.spatial.transform import Rotation
import numpy as np
from cv_bridge import CvBridge
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import time
import threading
import queue

# Franka DH parameters
FRANKA_DH = {
    'd': [0.333, 0, 0.316, 0, 0.384, 0, 0],
    'a': [0, 0, 0, 0.0825, -0.0825, 0, 0.088],
    'alpha': [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2],
    'offset': [0, 0, 0, 0, 0, 0, 0]
}

# xArm DH parameters
XARM_DH = {
    'd': [0.267, 0, 0.293, 0, 0.3425, 0, 0.076],
    'a': [0, 0, 0, 0, 0, 0, 0],
    'alpha': [-np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, 0],
    'offset': [0, 0, 0, 0, 0, 0, 0]
}

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
        
        self.gripper_action_client = SimpleActionClient(
            '/xarm/gripper_traj_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )
        if self.gripper_action_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo("Gripper trajectory controller connected")
        else:
            rospy.logwarn("Gripper trajectory controller not available")

        # Add gripper joint name
        self.gripper_joint_name = "drive_joint"  # Update this if different
        
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

        # Add queue for communication between threads
        # self.action_queue = queue.Queue(maxsize=1)
        
        # Start inference and execution threads
        # self.inference_thread = threading.Thread(target=self.inference_loop)
        # self.execution_thread = threading.Thread(target=self.execution_loop)
        
        # self.inference_thread.daemon = True
        # self.execution_thread.daemon = True
        
        if self.action_server_connected and self.policy_client is not None:
            self.main_control_loop()  # Instead of starting threads

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

    def send_xarm_gripper_command(self, target_position, duration=0.5):
        """
        Sends a command to the xArm gripper using trajectory controller.
        target_position: float, desired gripper position (0-0.85 meters)
        duration: float, time to reach target position in seconds
        """
        if not self.gripper_action_client.wait_for_server(rospy.Duration(1)):
            rospy.logwarn("Gripper trajectory controller not available")
            return

        # Create trajectory goal
        goal = FollowJointTrajectoryGoal()
        
        # Set joint names
        traj = JointTrajectory()
        traj.joint_names = [self.gripper_joint_name]
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [target_position]  # Single joint position
        point.velocities = [0.0]  # Zero velocity at goal
        point.time_from_start = rospy.Duration(duration)
        
        # Add point to trajectory
        traj.points.append(point)
        goal.trajectory = traj
        
        rospy.loginfo(f"Sending gripper command: position={target_position}")
        self.gripper_action_client.send_goal(goal)

    # def get_observation(self):
    #     """
    #     Returns observation with xArm images but Franka joint angles
    #     """
    #     if self.joint_positions is None:
    #         return None
        
    #     # Convert xArm joints to Franka end-effector pose
    #     xarm_pose, _ = self.compute_fk(self.joint_positions, XARM_DH)
        
    #     # Use IK to get corresponding Franka joints
    #     # This could be replaced with a more efficient mapping if needed
    #     franka_joints = self.compute_ik_franka(xarm_pose)
        
    #     return {
    #         "observation/exterior_image_1_left": 
    #             image_tools.resize_with_pad(self.camera_image_gazebo, 224, 224),
    #         "observation/wrist_image_left": 
    #             image_tools.resize_with_pad(self.camera_image_realsense_color, 224, 224),
    #         "observation/joint_position": franka_joints,  # Send Franka joints
    #         "observation/gripper_position": [self.gripper_pulse_state],
    #         "prompt": "Pick up the red marker and put it in the white bowl",
    #     }

    def get_observation_with_previous(self, previous_franka_joints):
        """
        Returns observation using previous Franka joints for better consistency
        """
        if self.joint_positions is None:
            return None
        
        if previous_franka_joints is None:
            # If no previous state, compute from current
            xarm_pose, _ = self.compute_fk(self.joint_positions, XARM_DH)
            franka_joints = self.compute_ik_franka(xarm_pose)
        else:
            franka_joints = previous_franka_joints
        
        return {
            "observation/exterior_image_1_left": 
                image_tools.resize_with_pad(self.camera_image_gazebo, 224, 224),
            "observation/wrist_image_left": 
                image_tools.resize_with_pad(self.camera_image_realsense_color, 224, 224),
            "observation/joint_position": franka_joints,
            "observation/gripper_position": [self.gripper_pulse_state],
            "prompt": "Pick up the red marker and put it in the white bowl",
        }

    # def inference_loop(self):
    #     rate = rospy.Rate(2)  # 2 Hz - one inference every 0.5 seconds
        
    #     while not rospy.is_shutdown():
    #         # Wait for sensor data - check if data exists and is valid
    #         if (self.camera_image_gazebo is None or 
    #             self.camera_image_realsense_color is None or 
    #             self.joint_positions is None):
    #             rate.sleep()
    #             continue
            
    #         # Get inference
    #         try:
    #             start_time = time.time()
    #             observation = self.get_observation()
    #             actions = self.policy_client.infer(observation)["actions"]
    #             end_time = time.time()
                
    #             rospy.loginfo(f"Inference time: {end_time - start_time:.3f}s")
                
    #             # Put new actions in queue, replace old ones if necessary
    #             if self.action_queue.full():
    #                 _ = self.action_queue.get_nowait()  # Remove old actions
    #             self.action_queue.put_nowait(actions)
                
    #         except Exception as e:
    #             rospy.logerr(f"Inference failed: {e}")
    
    #     rate.sleep()

    # def execution_loop(self):
    #     rate = rospy.Rate(15)  # 15 Hz
    #     current_actions = None
    #     action_index = 0
        
    #     while not rospy.is_shutdown():
    #         # Get new actions if needed
    #         if current_actions is None or action_index >= self.open_loop_horizon:
    #             try:
    #                 current_actions = self.action_queue.get_nowait()
    #                 action_index = 0
    #             except queue.Empty:
    #                 rospy.logwarn_throttle(1, "Waiting for actions...")
    #                 rate.sleep()
    #                 continue
            
    #         # Execute current action
    #         action_step = current_actions[action_index]
    #         trajectory, gripper_command = self.convert_action_to_trajectory_and_gripper(action_step)
            
    #         if trajectory is not None:
    #             self.send_trajectory(trajectory)
    #             self.send_xarm_gripper_command(gripper_command)
            
    #         action_index += 1
    #         rate.sleep()

    def convert_action_to_trajectory_and_gripper(self, action_step):
        if isinstance(action_step, list):
            action_step = np.array(action_step)
        
        dt = 1 / 15.0  # Time step in seconds
    
        if self.joint_positions is None:
            rospy.logerr("Current joint positions not available")
            return None, None
        
        # Convert current xArm state to Franka space for velocity integration
        xarm_pose, _ = self.compute_fk(self.joint_positions, XARM_DH)
        current_franka = self.compute_ik_franka(xarm_pose)
        
        # Integrate velocities in Franka space
        target_franka = current_franka + action_step[:7] * dt
        
        # Convert target Franka joints to xArm joints
        target_angles = self.franka_to_xarm_joints(target_franka)
        
        # Process gripper command
        gripper_val = action_step[7]
        gripper_command = np.clip(gripper_val, 0.0, 1.0) * 0.85
    
        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
    
        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(dt)
    
        traj_msg.points.append(point)
    
        return traj_msg, gripper_command

    def compute_fk(self, joint_angles, dh_params):
        """
        Compute forward kinematics using DH parameters
        """
        print("Computing FK:")
        T = np.eye(4)
        transforms = []
        
        for i in range(len(joint_angles)):
            theta = joint_angles[i] + dh_params['offset'][i]
            d = dh_params['d'][i]
            a = dh_params['a'][i]
            alpha = dh_params['alpha'][i]
            
            # DH transformation matrix
            T_i = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            
            T = T @ T_i
            transforms.append(T.copy())
        
        return T, transforms

    def compute_ik_xarm(self, target_pose, initial_guess=None):
        """
        Compute inverse kinematics for xArm using numerical optimization
        """

        print("Computing IK for xArm:")
        if initial_guess is None:
            initial_guess = np.zeros(7)
        
        def objective(q):
            current_pose, _ = self.compute_fk(q, XARM_DH)
            pose_error = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])
            rot_error = np.linalg.norm(current_pose[:3, :3] - target_pose[:3, :3], 'fro')
            return pose_error + 0.5 * rot_error
        
        from scipy.optimize import minimize
        
        # Joint limits for xArm
        bounds = [(-2*np.pi, 2*np.pi) for _ in range(7)]
        
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )

        print("the result of doing ik on xarm is ",result)
        
        return result.x if result.success else None

    def compute_ik_franka(self, target_pose, initial_guess=None):
        """
        Compute inverse kinematics for Franka directly using the target pose
        in the robot's local coordinate frame
        """
        if not isinstance(target_pose, np.ndarray):
            target_pose = np.array(target_pose)
        
        if initial_guess is None:
            initial_guess = np.zeros(7)
        
        def objective(q):
            current_pose, _ = self.compute_fk(q, FRANKA_DH)
            pose_error = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])
            rot_error = np.linalg.norm(current_pose[:3, :3] - target_pose[:3, :3], 'fro')
            return pose_error + 0.5 * rot_error
        
        bounds = [
            (-2.8973, 2.8973),  # Joint 1
            (-1.7628, 1.7628),  # Joint 2
            (-2.8973, 2.8973),  # Joint 3
            (-3.0718, -0.0698), # Joint 4
            (-2.8973, 2.8973),  # Joint 5
            (-0.0175, 3.7525),  # Joint 6
            (-2.8973, 2.8973)   # Joint 7
        ]
        
        from scipy.optimize import minimize
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-6}
        )
        
        if not result.success:
            rospy.logwarn("Franka IK failed to converge!")
            return None
            
        return result.x

    def franka_to_xarm_joints(self, franka_joints, initial_guess=None):
        """
        Convert Franka joint angles to xArm joint angles using FK/IK
        Args:
            franka_joints: array of Franka joint angles
            initial_guess: array of initial xArm joint angles for IK optimization
        """
        print("Converting Franka joints to xArm joints:")
        if franka_joints is None:
            rospy.logerr("Franka joint angles are None")
            return None
            
        # Get Franka end-effector pose
        franka_pose, _ = self.compute_fk(franka_joints, FRANKA_DH)
        
        # Use provided initial guess or fallback to current position
        if initial_guess is None:
            initial_guess = self.joint_positions
            
        # Compute xArm IK to match the pose
        xarm_joints = self.compute_ik_xarm(franka_pose, initial_guess=initial_guess)
        
        if xarm_joints is None:
            rospy.logwarn("IK failed to converge")
            return None
        
        return xarm_joints

    def move_to_position(self, target_joints):
        """
        Moves the robot arm to a specific joint configuration
        Args:
            target_joints: array of 7 joint angles for the arm
        """
        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names  # Uses arm joint names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = rospy.Duration(2.0)  # 2 second movement

        traj_msg.points.append(point)
        
        # Send to arm trajectory controller
        self.send_trajectory(traj_msg)
        rospy.sleep(2)  # Wait for movement to complete

    def main_control_loop(self):
        """
        Main control loop that handles inference and execution sequentially
        """
        rate = rospy.Rate(15)  # 15 Hz
        
        # Store previous states
        previous_franka_joints = None
        previous_xarm_joints = None
        
        # First move to default position and get initial Franka pose
        if self.action_server_connected:
            self.move_to_default_position()
            
            # Get initial Franka state from default xArm position
            xarm_pose, _ = self.compute_fk(self.joint_positions, XARM_DH)
            previous_franka_joints = self.compute_ik_franka(xarm_pose)
            previous_xarm_joints = self.joint_positions.copy()
            rospy.loginfo("Initial poses computed")
        
        while not rospy.is_shutdown():
            # Check if all required data is available
            if (self.camera_image_gazebo is None or 
                self.camera_image_realsense_color is None or 
                self.joint_positions is None):
                rospy.logwarn_throttle(1, "Waiting for sensor data...")
                rate.sleep()
                continue
                
            try:
                # 1. Get observation using previous Franka joints as reference
                observation = self.get_observation_with_previous(previous_franka_joints)
                
                # 2. Get policy actions
                start_time = time.time()
                actions = self.policy_client.infer(observation)["actions"]
                end_time = time.time()
                rospy.loginfo(f"Inference time: {end_time - start_time:.3f}s")
                
                # 3. Execute each action in the sequence
                for action_step in actions:
                    # Convert action to trajectory using previous states
                    trajectory, gripper_command = self.convert_action_with_previous(
                        action_step, 
                        previous_franka_joints,
                        previous_xarm_joints
                    )
                    
                    if trajectory is not None:
                        # Execute trajectory
                        self.send_trajectory(trajectory)
                        self.send_xarm_gripper_command(gripper_command)
                        
                        # Update previous states
                        xarm_pose, _ = self.compute_fk(trajectory.points[0].positions, XARM_DH)
                        previous_franka_joints = self.compute_ik_franka(
                            xarm_pose, 
                            initial_guess=previous_franka_joints
                        )
                        previous_xarm_joints = trajectory.points[0].positions
                        
                    rate.sleep()
                    
            except Exception as e:
                rospy.logerr(f"Control loop error: {e}")
                # On error, try to recover using last known good state
                if previous_xarm_joints is not None:
                    rospy.logwarn("Attempting recovery using last known position")
                    self.move_to_position(previous_xarm_joints)
                rate.sleep()
                continue
            
            rate.sleep()

    def convert_action_with_previous(self, action_step, prev_franka, prev_xarm):
        """
        Convert action to trajectory using previous states for better initialization
        """
        if isinstance(action_step, list):
            action_step = np.array(action_step)
        
        dt = 1 / 15.0  # Time step in seconds

        if prev_franka is None or prev_xarm is None:
            rospy.logerr("Previous joint states not available")
            return None, None
        
        # Integrate velocities in Franka space
        target_franka = prev_franka + action_step[:7] * dt
        
        # Convert target Franka joints to xArm joints using previous xArm state
        target_angles = self.franka_to_xarm_joints(
            target_franka, 
            initial_guess=prev_xarm
        )
        
        # Process gripper command
        gripper_val = action_step[7]
        gripper_command = np.clip(gripper_val, 0.0, 1.0) * 0.85

        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(dt)

        traj_msg.points.append(point)

        return traj_msg, gripper_command

if __name__ == '__main__':
    try:
        CameraJointControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass