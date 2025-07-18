#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from actionlib import SimpleActionClient
import numpy as np
from cv_bridge import CvBridge
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import time
# Import ikpy
import ikpy.chain

# A canonical default "ready" pose for the Franka robot
# This is used to seed the initial IK calculation to ensure the robot starts in a predictable,
# visually appropriate configuration. The values are for the 7 controllable joints.
FRANKA_CANONICAL_READY_POSE = [0, -0.785, 0, -2.356, 0, 1.57, 0.785]

# Note: ikpy uses a different joint representation (includes non-controllable links), so we find the active links.

XARM_POLICY_START_JOINT_POS = [-2.5, -0.6, 0.5, 1.2, 0.0, 1.2, 0.0]

class CameraJointControlNode:
    def __init__(self):
        # Define the joint names for the xArm
        self.joint_names = [
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7"
        ]

        rospy.init_node('camera_joint_control_node', anonymous=True)
        rospy.loginfo('camera_joint_control_node started')

        # --- Kinematics Setup (ikpy) ---
        rospy.loginfo("Loading robot models for ikpy...")
        try:
            self.franka_chain = ikpy.chain.Chain.from_urdf_file(
                "/home/gunjan/catkin_ws/panda_nomesh.urdf",
                active_links_mask=[False, True, True, True, True, True, True, True, False]
            )

            print("Number of joints in the Franka chain:", len(self.franka_chain.active_links_mask))
            print("Franka Joint names:", [j.name for j in self.franka_chain.links])

            self.xarm_chain = ikpy.chain.Chain.from_urdf_file(
                "/home/gunjan/catkin_ws/xarm7_nomesh.urdf",
                base_elements=["link_base"],   # Base of your robot
                active_links_mask=[False, True, True, True, True, True, True, True, False]
            ) 

            print("Number of joints in the xarm7 chain:", len(self.xarm_chain.active_links_mask))
            print("xarm7 Joint names:", [j.name for j in self.xarm_chain.links])
            
            self.franka_active_link_indices = self.franka_chain.active_links_mask
            self.xarm_active_link_indices = self.xarm_chain.active_links_mask

            rospy.loginfo("ikpy chains created successfully.")

            # --- Detailed ikpy Debugging ---
            rospy.loginfo("--- Franka Chain Inspection ---")
            for i, link in enumerate(self.franka_chain.links):
                joint_name = "N/A"
                joint_type = "no_joint"
                if hasattr(link, 'joint') and link.joint is not None:
                    joint_name = link.joint.name
                    joint_type = link.joint.joint_type
                rospy.loginfo(f"Link[{i}]: {link.name} | Joint: {joint_name} | Type: {joint_type}")
            rospy.loginfo(f"Franka Active Link Indices Detected: {self.franka_active_link_indices}")

            rospy.loginfo("--- xArm Chain Inspection ---")
            for i, link in enumerate(self.xarm_chain.links):
                joint_name = "N/A"
                joint_type = "no_joint"
                if hasattr(link, 'joint') and link.joint is not None:
                    joint_name = link.joint.name
                    joint_type = link.joint.joint_type
                rospy.loginfo(f"Link[{i}]: {link.name} | Joint: {joint_name} | Type: {joint_type}")
                
            print("Number of joints in the chain:", len(self.xarm_chain.active_links_mask))
            print("Joint names:", [j.name for j in self.xarm_chain.links])
            rospy.loginfo(f"xArm Active Link Indices Detected: {self.xarm_active_link_indices}")
            # --- End Detailed ikpy Debugging ---
        except Exception as e:
            rospy.logerr(f"Failed to load URDFs for ikpy: {e}")
            return

        # Initialize the bridge between ROS and OpenCV
        self.bridge = CvBridge()

        # --- Subscriptions and Clients ---
        self.camera_image_subscriber_gazebo = rospy.Subscriber(
            '/xarm_exterior_camera/image_raw', Image, self.camera_image_callback_gazebo, queue_size=10)
        self.joint_state_subscriber = rospy.Subscriber(
            '/xarm/joint_states', JointState, self.joint_state_callback, queue_size=10)
        self.action_client = SimpleActionClient(
            '/xarm/xarm7_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.action_server_connected = False
        if self.action_client.wait_for_server(rospy.Duration(15)):
            rospy.loginfo("xArm Joint Trajectory Action server connected")
            self.action_server_connected = True
        else:
            rospy.logerr("xArm Joint Trajectory Action server did not come up.")
        rospy.sleep(2)

        # --- Variables ---
        self.joint_positions = None
        self.camera_image_gazebo = None
        self.gripper_pulse_state = 0.0

        # --- Initialization ---
        if self.action_server_connected:
            rospy.loginfo("Waiting for initial joint state...")
            while self.joint_positions is None and not rospy.is_shutdown():
                rospy.sleep(0.1)

        # --- Websocket Client ---
        self.policy_client = None
        try:
            self.policy_client = WebsocketClientPolicy(host="10.4.25.44", port=8000)
            rospy.loginfo("Policy server connected")
        except Exception as e:
            rospy.logerr(f"Could not connect to policy server: {e}")

        if self.action_server_connected and self.policy_client is not None:
            # Initialize robot pose and get initial Franka equivalent joints
            initial_franka_joints = self.initialize_robot_pose()
            if initial_franka_joints is None:
                rospy.logerr("Failed to initialize robot pose. Exiting.")
                return
            self.main_control_loop(initial_franka_joints)

    # --- Callbacks ---
    def camera_image_callback_gazebo(self, msg):
        self.camera_image_gazebo = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    def joint_state_callback(self, msg):
        self.joint_positions = np.array([msg.position[msg.name.index(joint)] for joint in self.joint_names])

    # --- Action Methods ---
    def send_trajectory(self, trajectory):
        goal_msg = FollowJointTrajectoryGoal(trajectory=trajectory)
        if self.action_server_connected:
            self.action_client.send_goal(goal_msg)

            # Dynamically set the timeout based on the trajectory's duration + a buffer
            if not trajectory.points:
                rospy.logwarn("send_trajectory called with an empty trajectory.")
                return False
            
            # Get the duration of the last point in the trajectory
            trajectory_duration = trajectory.points[-1].time_from_start.to_sec()
            # Add a buffer (e.g., 2 seconds) to allow for execution overhead
            wait_duration = rospy.Duration(trajectory_duration + 2.0)

            rospy.loginfo(f"Waiting for trajectory to complete. Duration: {trajectory_duration:.2f}s, Timeout: {wait_duration.to_sec():.2f}s")

            if not self.action_client.wait_for_result(wait_duration):
                rospy.logwarn("Trajectory execution timed out.")
                return False
            
            result = self.action_client.get_result()
            if result and result.error_code == result.SUCCESSFUL:
                return True
            
            rospy.logwarn(f"Trajectory execution failed with error code: {result.error_code if result else 'N/A'}")
            return False
            
        rospy.logerr("Action server is down.")
        return False

    def get_observation(self, franka_joints):
        return {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(self.camera_image_gazebo, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(self.camera_image_gazebo, 224, 224), # Using gazebo image for both for now
            "observation/joint_position": franka_joints,
            "observation/gripper_position": [self.gripper_pulse_state],
            "prompt": "Pick up the red marker and put it in the white bowl",
        }

    def initialize_robot_pose(self):
        rospy.loginfo("Initializing robot pose for policy...")

        # Step 1: Move xArm directly to the policy start position
        rospy.loginfo("Moving xArm to policy start position...")
        target_xarm_joints = np.array(XARM_POLICY_START_JOINT_POS)
        traj_msg = JointTrajectory(
            joint_names=self.joint_names,
            points=[JointTrajectoryPoint(positions=target_xarm_joints, time_from_start=rospy.Duration(5.0))]
        )

        if not self.send_trajectory(traj_msg):
            rospy.logerr("Failed to move xArm to start position. Aborting initialization.")
            return None
        
        rospy.loginfo("Arrived at policy start position.")

        # Ensure joint_positions are updated. Give it a moment to be sure we have the latest state.
        rospy.sleep(0.5) 
        if self.joint_positions is None:
            rospy.logerr("Could not get xArm joint positions after moving to start pose.")
            return None

        rospy.loginfo(f"xArm is at start position: {self.joint_positions}")

        # Step 2: Calculate the equivalent Franka joint angles at this position
        rospy.loginfo("Calculating initial Franka joint configuration via IK...")

        # Get the current end-effector pose of the xArm
        current_xarm_ikpy = np.zeros(len(self.xarm_chain.links))
        current_xarm_ikpy[self.xarm_active_link_indices] = self.joint_positions
        xarm_pose = self.xarm_chain.forward_kinematics(current_xarm_ikpy)

        # Use the canonical ready pose as the initial guess for Franka's IK solver.
        # This guides the solver to a more natural and predictable solution.
        initial_franka_for_ik = np.zeros(len(self.franka_chain.links))
        initial_franka_for_ik[self.franka_active_link_indices] = FRANKA_CANONICAL_READY_POSE

        # Calculate the Franka joint angles that achieve the same pose
        initial_franka_joints = self.franka_chain.inverse_kinematics_frame(
            target=xarm_pose,
            initial_position=initial_franka_for_ik
        )

        if initial_franka_joints is None:
            rospy.logerr("Failed to compute initial Franka joint configuration via IK.")
            return None

        rospy.loginfo("Successfully calculated initial Franka joints.")
        return initial_franka_joints

    def main_control_loop(self, initial_franka_joints):
        rate = rospy.Rate(15)
        
        # Correctly initialize previous_franka_joints for ikpy
        previous_franka_joints = initial_franka_joints

        while not rospy.is_shutdown():
            if self.camera_image_gazebo is None or self.joint_positions is None:
                rospy.logwarn_throttle(1, "Waiting for sensor data...")
                rate.sleep()
                continue

            current_xarm_ikpy = np.zeros(len(self.xarm_chain.links))
            current_xarm_ikpy[self.xarm_active_link_indices] = self.joint_positions
            xarm_pose = self.xarm_chain.forward_kinematics(current_xarm_ikpy)
            
            current_franka_ikpy = self.franka_chain.inverse_kinematics_frame(target=xarm_pose, initial_position=previous_franka_joints)

            if current_franka_ikpy is None:
                rospy.logwarn("Could not compute current Franka configuration. Using last known good state.")
            else:
                previous_franka_joints = current_franka_ikpy

            try:
                observation = self.get_observation(previous_franka_joints[self.franka_active_link_indices])
                actions = self.policy_client.infer(observation)["actions"]
                
                for action_step in actions:
                    trajectory, gripper_command = self.convert_action(action_step, previous_franka_joints, self.joint_positions)
                    
                    if trajectory and self.send_trajectory(trajectory):
                        # self.send_xarm_gripper_command(gripper_command) # Gripper command sending can be added back if needed
                        self.gripper_pulse_state = gripper_command
                        
                        target_xarm_ikpy = np.zeros(len(self.xarm_chain.links))
                        target_xarm_ikpy[self.xarm_active_link_indices] = trajectory.points[0].positions
                        next_xarm_pose = self.xarm_chain.forward_kinematics(target_xarm_ikpy)
                        next_franka_joints = self.franka_chain.inverse_kinematics_frame(target=next_xarm_pose, initial_position=previous_franka_joints)

                        if next_franka_joints is not None:
                            previous_franka_joints = next_franka_joints
                    else:
                        rospy.logwarn("Trajectory execution failed or action conversion failed. Breaking from action sequence.")
                        break
                    rate.sleep()
            except Exception as e:
                rospy.logerr(f"Control loop error: {e}")
                rate.sleep()

    def convert_action(self, action_step, prev_franka_ikpy, prev_xarm_ros):
        rospy.loginfo(f"--- convert_action called ---")
        rospy.loginfo(f"Action Step: {action_step}")
        rospy.loginfo(f"Previous Franka IKPY: {prev_franka_ikpy}")
        rospy.loginfo(f"Previous xArm ROS: {prev_xarm_ros}")

        dt = 1 / 15.0
        
        action_ikpy = np.zeros(len(self.franka_chain.links))
        action_ikpy[self.franka_active_link_indices] = action_step[:7]
        
        target_franka_ikpy = prev_franka_ikpy + action_ikpy * dt
        rospy.loginfo(f"Target Franka IKPY (after action): {target_franka_ikpy}")

        target_franka_pose = self.franka_chain.forward_kinematics(target_franka_ikpy)
        rospy.loginfo(f"Target Franka Pose: {target_franka_pose}")
        
        initial_xarm_ikpy = np.zeros(len(self.xarm_chain.links))
        initial_xarm_ikpy[self.xarm_active_link_indices] = prev_xarm_ros
        rospy.loginfo(f"Initial xArm IKPY (seed for IK): {initial_xarm_ikpy}")

        target_angles_ikpy = self.xarm_chain.inverse_kinematics_frame(target=target_franka_pose, initial_position=initial_xarm_ikpy)
        
        if target_angles_ikpy is None:
            rospy.logwarn("IK for xArm returned None in convert_action.")
            return None, None
            
        target_angles_ros = target_angles_ikpy[self.xarm_active_link_indices]
        gripper_command = np.clip(action_step[7], 0.0, 1.0) * 0.85

        rospy.loginfo(f"Raw xArm IKPY output (convert_action): {target_angles_ikpy}")
        rospy.loginfo(f"Target xArm Joint Positions (convert_action): {target_angles_ros}")
        rospy.loginfo(f"Gripper Command: {gripper_command}")
        rospy.loginfo(f"--- End convert_action ---")

        traj_msg = JointTrajectory(joint_names=self.joint_names, points=[JointTrajectoryPoint(positions=target_angles_ros, time_from_start=rospy.Duration(dt))])
        return traj_msg, gripper_command

if __name__ == '__main__':
    try:
        CameraJointControlNode()
        rospy.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("Shutting down camera_joint_control_node.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
