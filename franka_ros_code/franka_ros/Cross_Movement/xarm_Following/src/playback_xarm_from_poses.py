#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import json
import os
import ikpy.chain
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from actionlib import SimpleActionClient
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans

class XArmIKPlayer:
    def __init__(self):
        """
        Initializes the XArmPlayer, using ikpy for kinematics and a direct action client for control.
        """
        rospy.init_node('xarm_ik_player', anonymous=True)

        # --- Joint and Controller Setup ---
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.joint_positions = None

        # Subscriber to get the current joint states
        self.joint_state_subscriber = rospy.Subscriber(
            '/xarm/joint_states', JointState, self.joint_state_callback, queue_size=1)

        rospy.loginfo("running the playback from poses")
        # Action client to send trajectories directly to the controller
        self.action_client = SimpleActionClient(
            '/xarm/xarm7_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for xArm Joint Trajectory Action server...")
        self.action_server_connected = False
        # Wait for the action server to come up
        # This will block until the server is available or timeout occurs
        if self.action_client.wait_for_server(rospy.Duration(30)):
            rospy.loginfo("xArm Joint Trajectory Action server connected")
            self.action_server_connected = True
        else:
            rospy.logerr("xArm Joint Trajectory Action server did not come up.")
        rospy.sleep(2)

        # --- Kinematics Setup (ikpy) ---
        rospy.loginfo("Loading xArm URDF for ikpy...")
        try:
            urdf_path = "/home/gunjan/catkin_ws/xarm7_nomesh.urdf"
            if not os.path.exists(urdf_path):
                raise IOError(f"URDF file not found at {urdf_path}")
            self.xarm_chain = ikpy.chain.Chain.from_urdf_file(
                urdf_path,
                base_elements=["link_base"],   # Base of your robot
                active_links_mask=[False, True, True, True, True, True, True, True, False]
            )
            rospy.loginfo("ikpy chain for xArm created successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load URDF for ikpy: {e}")
            sys.exit(1)

        # Wait until we get the first joint state message
        rospy.loginfo("Waiting for initial joint state...")
        while self.joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Initial joint state received.")

    def joint_state_callback(self, msg):
        """
        Callback to update the current joint positions of the robot.
        """
        try:
            # Ensure the order of joints matches self.joint_names
            positions = [msg.position[msg.name.index(j)] for j in self.joint_names]
            self.joint_positions = np.array(positions)
        except ValueError as e:
            rospy.logwarn_throttle(1.0, f"Could not find all joint names in joint_state message: {e}")


    def load_poses_as_matrices(self, filepath):
        """
        Loads poses, applies a two-stage correction (base and tool), and returns 4x4 matrices.
        """
        try:
            with open(filepath, 'r') as f:
                pose_list_raw = json.load(f)
            rospy.loginfo(f"Loaded {len(pose_list_raw)} raw poses from {filepath}")
        except IOError as e:
            rospy.logerr(f"Failed to read file {filepath}: {e}")
            return []

        # --- Stage 1: Base Orientation Correction (Wrist-to-Wrist) ---
        xarm_home_quat = np.array([0.0, 0.0, 1.0, 0.0])
        franka_home_quat = np.array([0.990, -0.131, 0.058, -0.010])
        xarm_home_rot = tf_trans.quaternion_matrix(xarm_home_quat)[:3, :3]
        franka_home_rot = tf_trans.quaternion_matrix(franka_home_quat)[:3, :3]
        base_correction_rot = np.dot(xarm_home_rot, franka_home_rot.T)
        rospy.loginfo("Applying base rotational correction.")

        # --- Stage 2: Tool-Center-Point (TCP) Correction ---
        # Transform from Franka wrist to Franka TCP
        T_franka_wrist_to_tcp = tf_trans.quaternion_matrix([0.0, 0.0, -0.383, 0.924])
        T_franka_wrist_to_tcp[0:3, 3] = [0.0, 0.0, 0.103]

        # Transform from xArm wrist to xArm TCP
        T_xarm_wrist_to_tcp = tf_trans.quaternion_matrix([0.0, 0.0, 0.0, 1.0])
        T_xarm_wrist_to_tcp[0:3, 3] = [0.0, 0.0, 0.172]

        # Correction: P_xarm_wrist = P_franka_wrist * T_franka_tcp * inv(T_xarm_tcp)
        tool_correction_matrix = np.dot(T_franka_wrist_to_tcp, np.linalg.inv(T_xarm_wrist_to_tcp))
        rospy.loginfo("Applying tool center point (TCP) correction.")

        target_matrices = []
        for p_raw in pose_list_raw:
            pos = p_raw['position']
            orient = p_raw['orientation']
            
            # Create a 4x4 matrix for the loaded Franka wrist pose
            loaded_pose_matrix = tf_trans.quaternion_matrix([orient['x'], orient['y'], orient['z'], orient['w']])
            loaded_pose_matrix[0:3, 3] = [pos['x'], pos['y'], pos['z']]

            # Apply base correction to rotation
            corrected_rot = np.dot(base_correction_rot, loaded_pose_matrix[:3, :3])
            base_corrected_pose = np.copy(loaded_pose_matrix)
            base_corrected_pose[:3, :3] = corrected_rot

            # Apply the final tool correction
            final_target_matrix = np.dot(base_corrected_pose, tool_correction_matrix)
            
            target_matrices.append(final_target_matrix)
            
        return target_matrices

    def execute_ik_trajectory(self, target_matrices):
        """
        Calculates a joint trajectory from a list of Cartesian poses using ikpy and executes it.
        """
        if not target_matrices:
            rospy.logerr("Cannot execute trajectory, list of target matrices is empty.")
            return

        rospy.loginfo("Starting IK calculation for the trajectory...")

        # Use the current robot position as the seed for the first IK calculation
        # ikpy expects a full joint array, including non-active links
        ik_seed = np.zeros(len(self.xarm_chain.links))
        ik_seed[self.xarm_chain.active_links_mask] = self.joint_positions
        
        joint_trajectory_points = []
        time_from_start = 0.0
        dt = 0.1  # Time step between points in seconds

        for i, target_matrix in enumerate(target_matrices):
            # Calculate IK for the current target pose, using the previous result as the seed
            target_joint_angles = self.xarm_chain.inverse_kinematics_frame(
                target=target_matrix,
                initial_position=ik_seed
            )
            
            if target_joint_angles is None:
                rospy.logwarn(f"IK failed for waypoint {i}. Skipping this point.")
                continue

            # Update the seed for the next iteration
            ik_seed = target_joint_angles

            # Extract the active joint values for the trajectory message
            ros_joint_angles = target_joint_angles[self.xarm_chain.active_links_mask]

            # Create the trajectory point
            time_from_start += dt
            point = JointTrajectoryPoint(
                positions=ros_joint_angles.tolist(),
                time_from_start=rospy.Duration.from_sec(time_from_start)
            )
            joint_trajectory_points.append(point)

        rospy.loginfo(f"IK calculation complete. Generated {len(joint_trajectory_points)} trajectory points.")

        if not joint_trajectory_points:
            rospy.logerr("No valid trajectory points were generated after IK. Aborting.")
            return

        # --- Send the Trajectory to the Robot ---
        traj_msg = JointTrajectory(
            joint_names=self.joint_names,
            points=joint_trajectory_points
        )

        goal_msg = FollowJointTrajectoryGoal(trajectory=traj_msg)
        rospy.loginfo("Sending trajectory to the action server...")
        self.action_client.send_goal(goal_msg)

        # Wait for the trajectory to finish
        wait_duration = rospy.Duration(time_from_start + 5.0) # Add a 5s buffer
        if self.action_client.wait_for_result(wait_duration):
            rospy.loginfo("Trajectory execution finished successfully.")
        else:
            rospy.logwarn("Trajectory execution timed out or was preempted.")

def main():
    try:
        player = XArmIKPlayer()

        # Path to the recorded poses file
        poses_filepath = "/home/gunjan/catkin_ws/src/franka_ros_code/franka_ros/Cross_Movement/Franka_Recording/franka_poses.json"

        # Load poses as a list of 4x4 matrices
        target_matrices = player.load_poses_as_matrices(poses_filepath)

        if not target_matrices:
            rospy.logerr("Aborting due to empty pose list.")
            return

        # Execute the trajectory using IK
        rospy.loginfo("--- Starting xArm IK Playback ---")
        player.execute_ik_trajectory(target_matrices)
        rospy.loginfo("Playback finished.")

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"An unhandled error occurred in main: {e}")
    finally:
        rospy.loginfo("Shutting down xArm IK player.")

if __name__ == '__main__':
    main()
