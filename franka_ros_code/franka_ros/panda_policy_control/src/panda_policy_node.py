#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, GripperCommandAction, GripperCommandGoal, JointTolerance
from actionlib import SimpleActionClient
import numpy as np
from cv_bridge import CvBridge
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import time

class CameraJointControlNode:
    def __init__(self):
        rospy.init_node('camera_joint_control_node', anonymous=True)
        rospy.loginfo("Franka Panda camera joint control node started")

        # Initialize the bridge between ROS and OpenCV
        self.bridge = CvBridge()
        self.open_loop_horizon = 10  # Number of actions to execute before fetching new inference

        # --- Camera Subscriptions ---
        self.top_view_sub = rospy.Subscriber(
            '/panda_camera1/image_raw', Image, self.top_view_callback, queue_size=10)
        self.gripper_cam_sub = rospy.Subscriber(
            '/panda_camera2/image_raw', Image, self.gripper_camera_callback, queue_size=10)

        # --- Joint State Subscription ---
        self.joint_state_sub = rospy.Subscriber(
            '/franka_state_controller/joint_states', JointState, self.joint_state_callback, queue_size=10)

        # --- Gripper Joint State Subscription ---
        self.gripper_joint_state_sub = rospy.Subscriber(
            '/franka_gripper/joint_states', JointState, self.gripper_joint_state_callback, queue_size=10)

        # --- Action Client (Trajectory Execution) ---
        self.action_client = SimpleActionClient(
            '/position_joint_trajectory_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction)

        self.gripper_action_client = SimpleActionClient(
            '/franka_gripper/gripper_action', 
            GripperCommandAction)

        self.action_server_connected = False
        if self.action_client.wait_for_server(rospy.Duration(30)):
            rospy.loginfo("Action server connected")
            self.action_server_connected = True
        else:
            rospy.logerr("Action server did not come up, not sending goals.")
        rospy.sleep(2)

        # --- Variables to store data ---
        self.joint_positions = None
        self.top_view_image = None
        self.gripper_image = None
        self.finger_joint1 = 0.0
        self.finger_joint2 = 0.0

        # Joint names for Franka Panda (arm only)
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", 
            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        ]

        # --- Define Joint Limits (in radians) for Franka Panda ---
        self.franka_joint_limits = {
            "panda_joint1": (-2.8973, 2.8973),
            "panda_joint2": (-1.7628, 1.7628),
            "panda_joint3": (-2.8973, 2.8973),
            "panda_joint4": (-3.0718, -0.0698),
            "panda_joint5": (-2.8973, 2.8973),
            "panda_joint6": (-0.0175, 3.7525),
            "panda_joint7": (-2.8973, 2.8973)
        }

        # --- Define Velocity Limits for Franka Panda (in rad/s) ---
        self.franka_velocity_limits = {
            "panda_joint1": 2.1750,
            "panda_joint2": 2.1750,
            "panda_joint3": 2.1750,
            "panda_joint4": 2.1750,
            "panda_joint5": 2.61,
            "panda_joint6": 2.61,
            "panda_joint7": 2.61
        }

        # --- Counters for violations ---
        self.joint_limit_violation_count = 0
        self.self_collision_count = 0
        self.velocity_violation_count = 0
        self.singularity_count = 0

        # Record task start time for total execution time calculation
        self.task_start_time = time.time()

        # Move the robot to a default home position if the action server is connected.
        if self.action_server_connected:
            self.move_to_default_position()

        # --- Websocket Policy Client ---
        try:
            self.policy_client = WebsocketClientPolicy(host="10.4.25.44", port=8000)
            rospy.loginfo("Policy server connected")
        except Exception as e:
            rospy.logerr(f"Could not connect to policy server: {e}")
            self.policy_client = None

        # Start control loop only if both connections are valid.
        if self.action_server_connected and self.policy_client is not None:
            rospy.loginfo("Starting control loop")
            self.control_loop()

    # --- Camera Callbacks ---
    def top_view_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.top_view_image = cv_image

    def gripper_camera_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.gripper_image = cv_image

    # --- Joint State Callback ---
    def joint_state_callback(self, msg):
        self.joint_positions = np.array(msg.position)

    # --- Gripper Joint State Callback ---
    def gripper_joint_state_callback(self, msg):
        if len(msg.position) >= 2:
            self.finger_joint1 = msg.position[0]
            self.finger_joint2 = msg.position[1]

    # --- Action Client Methods ---
    def send_trajectory(self, trajectory):
        goal_msg = FollowJointTrajectoryGoal()
        goal_msg.trajectory = trajectory

        # Add tolerances for each joint
        for joint_name in self.joint_names:
            tolerance = JointTolerance()
            tolerance.name = joint_name
            tolerance.position = rospy.get_param(f"/{joint_name}/goal_tolerance", 0.05)
            goal_msg.goal_tolerance.append(tolerance)

        rospy.loginfo("Sending joint trajectory")
        if self.action_server_connected:
            self.action_client.send_goal(goal_msg)
            self.action_client.wait_for_result()
            result = self.action_client.get_result()
            if result:
                rospy.loginfo(f"Joint trajectory result: {result}")
                rospy.loginfo(f"Current joint positions: {self.joint_positions}")
            else:
                rospy.logerr("Failed to execute joint trajectory")
        else:
            rospy.logerr("Joint action server is down, could not send goal")

    def send_gripper_command(self, finger_joint1, finger_joint2):
        # Clamp gripper commands to valid range
        finger_joint1 = max(0.0, min(finger_joint1, 0.04))
        finger_joint2 = max(0.0, min(finger_joint2, 0.04))
        goal_msg = GripperCommandGoal()
        goal_msg.command.position = (finger_joint1 + finger_joint2) / 2.0
        goal_msg.command.max_effort = 10.0
        rospy.loginfo(f"Sending gripper command: position={goal_msg.command.position}")
        if self.gripper_action_client.wait_for_server(rospy.Duration(5)):
            self.gripper_action_client.send_goal(goal_msg)
            self.gripper_action_client.wait_for_result()
            result = self.gripper_action_client.get_result()
            if result:
                rospy.loginfo(f"Gripper command result: {result}")
                rospy.loginfo(f"Current gripper position: {goal_msg.command.position}")
            else:
                rospy.logerr("Failed to execute gripper command")
        else:
            rospy.logerr("Gripper action server is down, could not send goal")

    def move_to_default_position(self):
        rospy.loginfo("Moving to default (home) position...")
        # Default joint angles for Franka Panda (in radians)
        # This home position is chosen as a safe starting position (similar to those in the DROID Dataset).
        target_angles = [-0.1, -0.8, 0.7, -2.356194, 0.0, 1.5708, 0.085398]

        gripper_position = [0.04, 0.04]  # Open gripper

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(3.0)
        rospy.loginfo(f"Default joint positions: {point.positions}")

        traj_msg.points.append(point)
        self.send_trajectory(traj_msg)
        self.send_gripper_command(gripper_position[0], gripper_position[1])
        rospy.sleep(5)
        rospy.loginfo("Arrived at default position")

    # --- Helper Functions for Checks ---
    def check_joint_limits(self, target_angles):
        violation_count = 0
        modified_angles = []
        for joint_name, angle in zip(self.joint_names, target_angles):
            if joint_name in self.franka_joint_limits:
                (min_angle, max_angle) = self.franka_joint_limits[joint_name]
                if angle < min_angle:
                    violation_count += 1
                    rospy.logwarn(f"{joint_name} angle {angle:.3f} below minimum {min_angle:.3f}, clipping.")
                    angle = min_angle
                elif angle > max_angle:
                    violation_count += 1
                    rospy.logwarn(f"{joint_name} angle {angle:.3f} above maximum {max_angle:.3f}, clipping.")
                    angle = max_angle
            modified_angles.append(angle)
        self.joint_limit_violation_count += violation_count
        return modified_angles
    
    def check_velocity_limit(self, action_velocities):
            violation_count = 0
            # Scale factor should match the one used in convert_action_to_trajectory_and_gripper
            scale_factor = 1.0
            
            for joint_name, velocity in zip(self.joint_names, action_velocities[:7]):
                # Scale the normalized action velocity to actual velocity
                actual_velocity = abs(velocity * scale_factor)
                if joint_name in self.franka_velocity_limits:
                    v_limit = self.franka_velocity_limits[joint_name]
                    if actual_velocity > 0.8 * v_limit:
                        violation_count += 1
                        rospy.logwarn(f"Joint {joint_name} velocity {actual_velocity:.3f} exceeds 80% of limit {0.8*v_limit:.3f}")
            
            self.velocity_violation_count += violation_count
            return violation_count

    def check_self_collision(self, joint_angles):
        # integrate with a collision-checking library.
        collision = False
        if collision:
            self.self_collision_count += 1
        return collision
    
    def get_transformation_matrix(self,a, d, alpha, theta):
        """
        Returns the 4x4 homogeneous transformation matrix using
        standard Denavit-Hartenberg parameters.
        """
        return np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  d*np.cos(alpha)],
        [0,               0,                0,             1]
        ], dtype=float)

    def compute_jacobian(self, joint_angles):
        """
        Computes the 6x7 Jacobian matrix for the Franka Emika Panda
        using the given joint_angles.

        joint_angles: array-like of length 7
        returns: a 6x7 NumPy array (the Jacobian)
        """

        # DH parameters for Franka Panda (8th row for flange is included to compute final T_EE)
        # Each row: [a, d, alpha, theta]
        dh_params = np.array([
            [0.0,    0.333,    0.0,         joint_angles[0]],
            [0.0,    0.0,     -np.pi/2,     joint_angles[1]],
            [0.0,    0.316,    np.pi/2,     joint_angles[2]],
            [0.0825, 0.0,      np.pi/2,     joint_angles[3]],
            [-0.0825,0.384,   -np.pi/2,     joint_angles[4]],
            [0.0,    0.0,      np.pi/2,     joint_angles[5]],
            [0.088,  0.0,      np.pi/2,     joint_angles[6]],
            [0.0,    0.107,    0.0,         0.0]  # flange (fixed)
        ], dtype=float)

        # Compute the final transformation to end effector
        T_EE = np.eye(4)
        for i in range(dh_params.shape[0]):
            a, d, alpha, theta = dh_params[i]
            T_EE = np.dot(T_EE, self.get_transformation_matrix(a, d, alpha, theta))


        # Initialize 6x8 Jacobian (for 7 joints + 1 flange link),
        # but we'll only need the first 7 columns for the joints.
        J = np.zeros((6, dh_params.shape[0]))

        # Compute incremental transformations for each link and fill in Jacobian columns
        T_current = np.eye(4)
        for i in range(dh_params.shape[0]):
            # Extract transformation up to the i-th link
            a, d, alpha, theta = dh_params[i]
            T_current = np.dot(T_current, self.get_transformation_matrix(a, d, alpha, theta))
            # Position of end-effector relative to i-th frame
            p = T_EE[:3, 3] - T_current[:3, 3]
            # z-axis of the i-th joint frame
            z = T_current[:3, 2]

            # Linear part (cross product of z and p)
            J[:3, i] = np.cross(z, p)
            # Angular part (just z)
            J[3:, i] = z

        # Return only the first 7 columns corresponding to the 7 joints
        return J[:, :7]

    def check_singularity(self, joint_angles, singularity_threshold=0.01):
        """
        Checks if the configuration defined by joint_angles is near a singularity.
        
        Parameters:
            joint_angles (list or np.array): Joint angles of the robot.
            singularity_threshold (float): Threshold for the smallest singular value.
        
        Returns:
            bool: True if the configuration is near a singularity, False otherwise.
        """
        # Compute the Jacobian matrix for the current joint angles.
        J = self.compute_jacobian(joint_angles)
        
        # Perform Singular Value Decomposition on the Jacobian.
        U, singular_values, Vh = np.linalg.svd(J)
        
        # For a non-singular configuration, the smallest singular value should be above the threshold.
        min_singular_value = singular_values[-1]
        
        # Debug: Print the singular values to observe their values.
        # print("Singular values:", singular_values)
        
        if min_singular_value < singularity_threshold:
            print(f"Near singularity detected: smallest singular value = {min_singular_value:.6f}")
            return True
        else:
            return False

    # --- Main Control Loop ---
    def control_loop(self):
        rate = rospy.Rate(15)  # 15 Hz control loop
        action_chunk = None

        while not rospy.is_shutdown():
            # Ensure images are available
            if self.top_view_image is None:
                rospy.logwarn("No image received from /panda_camera1/image_raw yet.")
                rate.sleep()
                continue

            if self.gripper_image is None:
                rospy.logwarn("No image received from /panda_camera2/image_raw yet.")
                rate.sleep()
                continue

            # Prepare observation for the policy
            observation = {
                "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(self.top_view_image, 224, 224)
                ),
                "observation/wrist_image_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(self.gripper_image, 224, 224)
                ),
                "observation/joint_position": self.joint_positions,
                "observation/gripper_position": [(self.finger_joint1 + self.finger_joint2) / 2],
                "prompt": "Pick up the marker and put it in the bowl"
            }

            if self.policy_client is not None and action_chunk is None:
                start_inference = time.time()
                action_chunk = self.policy_client.infer(observation)["actions"]
                end_inference = time.time()
                # Round each action (inner list) to 3 decimal places
                action_chunk = [np.round(action, 3).tolist() for action in action_chunk]
                rospy.loginfo(f"Inference time: {end_inference - start_inference:.3f} seconds")

            if action_chunk is not None:
                rospy.loginfo(f"Processing action_chunk: shape {len(action_chunk), len(action_chunk[0])}")
                # Only execute up to open_loop_horizon actions
                for idx in range(self.open_loop_horizon):
                    action = action_chunk[idx]
                    rospy.loginfo(f"Processing action index: {idx}")
                    # log current action chunk
                    rospy.loginfo(f"Action: {action}")
                    action = np.clip(action, -1, 1)
                    traj_and_gripper = self.convert_action_to_trajectory_and_gripper(action)
                    if traj_and_gripper is None:
                        rospy.logerr("Conversion of action to trajectory failed.")
                        continue
                    trajectory, gripper_command = traj_and_gripper
                    rospy.loginfo(f"Trajectory: {trajectory.points[0].positions}")
                    rospy.loginfo(f"Gripper command: {gripper_command}")
                    self.send_trajectory(trajectory)
                    self.send_gripper_command(gripper_command[0], gripper_command[1])
                    rate.sleep()  # Wait for next step
                # Once done, report total execution time and counters
                total_time = time.time() - self.task_start_time
                rospy.loginfo(f"Total command execution time: {total_time:.3f} seconds")
                rospy.loginfo(f"Joint limit violations: {self.joint_limit_violation_count}")
                rospy.loginfo(f"Velocity limit violations: {self.velocity_violation_count}")
                rospy.loginfo(f"Self collision count: {self.self_collision_count}")
                rospy.loginfo(f"Singularity warnings: {self.singularity_count}")
                action_chunk = None  
            else:
                rospy.logwarn("No valid action chunk available, skipping execution")
                rate.sleep()

    def convert_action_to_trajectory_and_gripper(self, action_step):
        """
        Converts a single action step (interpreted as joint velocities) into a JointTrajectory
        and gripper command. This integrates the velocities over a fixed time step (dt) to update
        the current joint positions. It also performs checking for joint limits, velocity limits,
        self-collision, and singularity.
        """
        # print("Action step:", action_step)
        if isinstance(action_step, list):
            action_step = np.array(action_step)

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names  # Arm joints only

        gripper_command = [0.0, 0.0]  # Default gripper command

        dt = 1/15  # Integration time step (seconds) changed because each action should take approx 1/15 seconds

        if self.joint_positions is None:
            rospy.logerr("Current joint positions not available. Cannot integrate velocities.")
            return None

        # Integrate the velocities into positions:
        # new_position = current_position + (velocity * dt)
        current_angles = self.joint_positions[:7]
        scale_factor = 1.0  # Experiment with different values
        target_angles_rad = current_angles + scale_factor * action_step[:7] * dt
        
        target_angles = np.degrees(target_angles_rad)


        # Check joint limits and clip if necessary
        target_angles = self.check_joint_limits(target_angles)

        # Check for velocity limit violations (using action velocities directly)
        self.check_velocity_limit(action_step[:7])
        # self.check_velocity_limit(current_angles, target_angles, dt)

        # Check for self-collision (replace with actual collision-check)
        if self.check_self_collision(target_angles):
            rospy.logwarn("Self collision detected for target angles!")
            # Increase counter already in check_self_collision()

        # Check for singularity (simple heuristic)
        if self.check_singularity(target_angles):
            rospy.logwarn("Singularity condition reached in target angles!")

        traj_msg.points = []
        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(dt)
        traj_msg.points.append(point)

        # Process gripper command (last value of action)
        gripper_position = action_step[7]
        if gripper_position < 0.0:
            gripper_command = [0.0, 0.0]
        elif gripper_position > 1.0:
            gripper_command = [0.04, 0.04]
        else:
            gripper_command = [gripper_position * 0.04, gripper_position * 0.04]

        return traj_msg, gripper_command

if __name__ == '__main__':
    try:
        node = CameraJointControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
