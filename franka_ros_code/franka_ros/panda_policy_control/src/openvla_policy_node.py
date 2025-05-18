#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, GripperCommandAction, GripperCommandGoal, JointTolerance
from actionlib import SimpleActionClient
import numpy as np
import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage
import time
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import math

class OpenVLARobotControlNode:
    def __init__(self):
        rospy.init_node('openvla_control_node', anonymous=True)
        rospy.loginfo("OpenVLA control node started")

        # Initialize CV Bridge and variables for storing sensor data
        self.bridge = CvBridge()
        self.camera_image = None
        self.joint_positions = None
        self.finger_joint1 = 0.0
        self.finger_joint2 = 0.0

        # --- Subscribers ---
        self.camera_sub = rospy.Subscriber('/panda_camera1/image_raw', Image, self.camera_callback, queue_size=10)
        self.joint_state_sub = rospy.Subscriber('/franka_state_controller/joint_states', JointState, self.joint_state_callback, queue_size=10)
        self.gripper_joint_state_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, self.gripper_joint_state_callback, queue_size=10)

        # --- Action Clients ---
        self.action_client = SimpleActionClient('/position_joint_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.gripper_action_client = SimpleActionClient('/franka_gripper/gripper_action', GripperCommandAction)
        self.action_server_connected = False
        if self.action_client.wait_for_server(rospy.Duration(30)):
            rospy.loginfo("Joint trajectory action server connected")
            self.action_server_connected = True
        else:
            rospy.logerr("Joint trajectory action server did not come up, not sending goals.")
        rospy.sleep(2)

        # --- Robot Configuration ---
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        self.franka_joint_limits = {
            "panda_joint1": (-2.8973, 2.8973),
            "panda_joint2": (-1.7628, 1.7628),
            "panda_joint3": (-2.8973, 2.8973),
            "panda_joint4": (-3.0718, -0.0698),
            "panda_joint5": (-2.8973, 2.8973),
            "panda_joint6": (-0.0175, 3.7525),
            "panda_joint7": (-2.8973, 2.8973)
        }
        self.franka_velocity_limits = {
            "panda_joint1": 2.1750,
            "panda_joint2": 2.1750,
            "panda_joint3": 2.1750,
            "panda_joint4": 2.1750,
            "panda_joint5": 2.61,
            "panda_joint6": 2.61,
            "panda_joint7": 2.61
        }
        self.joint_limit_violation_count = 0
        self.velocity_violation_count = 0
        self.self_collision_count = 0
        self.singularity_count = 0

        # For this example, we execute one action per inference.
        self.open_loop_horizon = 1

        self.inference_server_url = "http://your_gpu_server_ip:5000/predict"
        rospy.loginfo("Using inference server at: %s", self.inference_server_url)

        # --- Move to Default Position ---
        if self.action_server_connected:
            self.move_to_default_position()

        # --- Start Control Loop ---
        self.control_loop()

    # --- Callback Functions ---
    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.camera_image = cv_image
        except Exception as e:
            rospy.logerr("Error in camera_callback: %s", e)

    def joint_state_callback(self, msg):
        self.joint_positions = np.array(msg.position)

    def gripper_joint_state_callback(self, msg):
        if len(msg.position) >= 2:
            self.finger_joint1 = msg.position[0]
            self.finger_joint2 = msg.position[1]

    # --- Default Position ---
    def move_to_default_position(self):
        rospy.loginfo("Moving to default (home) position...")
        target_angles = [-0.1, -0.8, 0.7, -2.356194, 0.0, 1.5708, 0.085398]
        gripper_position = [0.04, 0.04]  # Open gripper
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(3.0)
        traj_msg.points.append(point)
        self.send_trajectory(traj_msg)
        self.send_gripper_command(gripper_position[0], gripper_position[1])
        rospy.sleep(5)
        rospy.loginfo("Arrived at default position.")

    # --- Action Server Calls ---
    def send_trajectory(self, trajectory):
        goal_msg = FollowJointTrajectoryGoal()
        goal_msg.trajectory = trajectory
        for joint_name in self.joint_names:
            tol = JointTolerance()
            tol.name = joint_name
            tol.position = rospy.get_param(f"/{joint_name}/goal_tolerance", 0.05)
            goal_msg.goal_tolerance.append(tol)
        rospy.loginfo("Sending joint trajectory")
        if self.action_server_connected:
            self.action_client.send_goal(goal_msg)
            self.action_client.wait_for_result()
            result = self.action_client.get_result()
            if result:
                rospy.loginfo("Trajectory execution result: %s", result)
            else:
                rospy.logerr("Failed to execute joint trajectory")
        else:
            rospy.logerr("Joint action server is down, cannot send goal")

    def send_gripper_command(self, finger_joint1, finger_joint2):
        # Clamp gripper commands to valid range
        finger_joint1 = max(0.0, min(finger_joint1, 0.04))
        finger_joint2 = max(0.0, min(finger_joint2, 0.04))
        goal_msg = GripperCommandGoal()
        goal_msg.command.position = (finger_joint1 + finger_joint2) / 2.0
        goal_msg.command.max_effort = 10.0
        rospy.loginfo("Sending gripper command: position = %s", goal_msg.command.position)
        if self.gripper_action_client.wait_for_server(rospy.Duration(5)):
            self.gripper_action_client.send_goal(goal_msg)
            self.gripper_action_client.wait_for_result()
            result = self.gripper_action_client.get_result()
            if result:
                rospy.loginfo("Gripper command result: %s", result)
            else:
                rospy.logerr("Failed to execute gripper command")
        else:
            rospy.logerr("Gripper action server is down, cannot send goal")

    # --- Helper Functions ---
    def check_joint_limits(self, target_angles):
        violation_count = 0
        modified_angles = []
        for joint_name, angle in zip(self.joint_names, target_angles):
            if joint_name in self.franka_joint_limits:
                (min_angle, max_angle) = self.franka_joint_limits[joint_name]
                if angle < min_angle:
                    violation_count += 1
                    rospy.logwarn("%s angle %f below minimum %f, clipping", joint_name, angle, min_angle)
                    angle = min_angle
                elif angle > max_angle:
                    violation_count += 1
                    rospy.logwarn("%s angle %f above maximum %f, clipping", joint_name, angle, max_angle)
                    angle = max_angle
            modified_angles.append(angle)
        self.joint_limit_violation_count += violation_count
        return modified_angles

    def check_velocity_limit(self, action_velocities):
        violation_count = 0
        scale_factor = 1.0
        for joint_name, velocity in zip(self.joint_names, action_velocities[:7]):
            actual_velocity = abs(velocity * scale_factor)
            if joint_name in self.franka_velocity_limits:
                v_limit = self.franka_velocity_limits[joint_name]
                if actual_velocity > 0.8 * v_limit:
                    violation_count += 1
                    rospy.logwarn("Joint %s velocity %f exceeds 80%% of limit %f", joint_name, actual_velocity, 0.8 * v_limit)
        self.velocity_violation_count += violation_count
        return violation_count

    def check_self_collision(self, joint_angles):
        collision = False
        # Example: flag collision if any joint angle is near its limit.
        for joint_name, angle in zip(self.joint_names, joint_angles):
            (min_angle, max_angle) = self.franka_joint_limits[joint_name]
            if abs(angle - min_angle) < 0.01 or abs(angle - max_angle) < 0.01:
                collision = True
                rospy.logwarn("Self collision warning for joint %s", joint_name)
        if collision:
            self.self_collision_count += 1
        return collision

    def get_transformation_matrix(self, a, d, alpha, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha),  np.cos(alpha),  d * np.cos(alpha)],
            [0, 0, 0, 1]
        ], dtype=float)

    def compute_jacobian(self, joint_angles):
        dh_params = np.array([
            [0.0,    0.333,    0.0,         joint_angles[0]],
            [0.0,    0.0,     -np.pi/2,     joint_angles[1]],
            [0.0,    0.316,    np.pi/2,     joint_angles[2]],
            [0.0825, 0.0,      np.pi/2,     joint_angles[3]],
            [-0.0825,0.384,   -np.pi/2,     joint_angles[4]],
            [0.0,    0.0,      np.pi/2,     joint_angles[5]],
            [0.088,  0.0,      np.pi/2,     joint_angles[6]],
            [0.0,    0.107,    0.0,         0.0]  # flange
        ], dtype=float)

        T_EE = np.eye(4)
        for i in range(dh_params.shape[0]):
            a, d, alpha, theta = dh_params[i]
            T_EE = np.dot(T_EE, self.get_transformation_matrix(a, d, alpha, theta))

        J = np.zeros((6, dh_params.shape[0]))
        T_current = np.eye(4)
        for i in range(dh_params.shape[0]):
            a, d, alpha, theta = dh_params[i]
            T_current = np.dot(T_current, self.get_transformation_matrix(a, d, alpha, theta))
            p = T_EE[:3, 3] - T_current[:3, 3]
            z = T_current[:3, 2]
            J[:3, i] = np.cross(z, p)
            J[3:, i] = z
        return J[:, :7]

    def check_singularity(self, joint_angles, singularity_threshold=0.01):
        J = self.compute_jacobian(joint_angles)
        _, singular_values, _ = np.linalg.svd(J)
        min_singular_value = singular_values[-1]
        if min_singular_value < singularity_threshold:
            rospy.logwarn("Near singularity detected: smallest singular value = %f", min_singular_value)
            self.singularity_count += 1
            return True
        return False

    def convert_action_to_trajectory_and_gripper(self, action_step):
        """
        Converts a single action step (expected as an array of 8 values: 7 for joint deltas,
        and 1 for gripper) into a JointTrajectory message and a gripper command.
        """
        if isinstance(action_step, list):
            action_step = np.array(action_step)
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        gripper_command = [0.0, 0.0]
        dt = 1.0 / 15.0  # Time step (s)
        if self.joint_positions is None:
            rospy.logerr("Current joint positions not available.")
            return None
        current_angles = self.joint_positions[:7]
        scale_factor = 1.0  # Adjust based on how the model output is normalized
        target_angles = current_angles + scale_factor * action_step[:7] * dt
        target_angles = self.check_joint_limits(target_angles)
        self.check_velocity_limit(action_step[:7])
        if self.check_self_collision(target_angles):
            rospy.logwarn("Self collision detected for target angles!")
        if self.check_singularity(target_angles):
            rospy.logwarn("Singularity condition reached in target angles!")
        point = JointTrajectoryPoint()
        point.positions = target_angles
        point.time_from_start = rospy.Duration(dt)
        traj_msg.points.append(point)
        # Process gripper command (if provided as the 8th value)
        gripper_position = action_step[7] if len(action_step) > 7 else 0.0
        if gripper_position < 0.0:
            gripper_command = [0.0, 0.0]
        elif gripper_position > 1.0:
            gripper_command = [0.04, 0.04]
        else:
            gripper_command = [gripper_position * 0.04, gripper_position * 0.04]
        return traj_msg, gripper_command

    def run_inference(self):
        """
        Sends the camera image to remote server for inference
        """
        if self.camera_image is None:
            rospy.logwarn("No camera image available for inference.")
            return None

        try:
            # Convert OpenCV image to RGB
            cv_image_rgb = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(cv_image_rgb)
            
            # Convert PIL image to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Send to server
            response = requests.post(
                self.inference_server_url,
                json={'image': img_str},
                timeout=10
            )
            
            if response.status_code == 200:
                action = response.json()['action']
                return np.array(action)
            else:
                rospy.logerr(f"Server error: {response.status_code}")
                return None
                
        except Exception as e:
            rospy.logerr(f"Error in inference: {str(e)}")
            return None

    def control_loop(self):
        rate = rospy.Rate(15)  # 15 Hz control loop
        while not rospy.is_shutdown():
            if self.camera_image is None:
                rospy.logwarn("Waiting for camera image...")
                rate.sleep()
                continue
            action = self.run_inference()
            if action is not None:
                rospy.loginfo("Predicted action: %s", action)
                result = self.convert_action_to_trajectory_and_gripper(action)
                if result is not None:
                    traj, gripper_command = result
                    rospy.loginfo("Trajectory joint positions: %s", traj.points[0].positions)
                    rospy.loginfo("Gripper command: %s", gripper_command)
                    self.send_trajectory(traj)
                    self.send_gripper_command(gripper_command[0], gripper_command[1])
            rate.sleep()

if __name__ == '__main__':
    try:
        node = OpenVLARobotControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
