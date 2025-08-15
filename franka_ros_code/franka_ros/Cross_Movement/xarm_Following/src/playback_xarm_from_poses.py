#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import json
import os
import ikpy.chain
import tf
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

        # --- TF Listener ---
        self.tf_listener = tf.TransformListener()

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

        # To store the recorded poses of the xArm
        self.recorded_xarm_poses = []

        # --- Kinematics Setup (ikpy) ---
        rospy.loginfo("Loading xArm URDF for ikpy...")
        try:
            urdf_path = "/home/gunjan/catkin_ws/xarm7_ik_clean.urdf"
            if not os.path.exists(urdf_path):
                raise IOError(f"URDF file not found at {urdf_path}")
            self.xarm_chain = ikpy.chain.Chain.from_urdf_file(
                urdf_path,
                base_elements=["link_base"],   # Base of your robot
                active_links_mask=[False, True, True, True, True, True, True, True, False]
            )
            rospy.loginfo("ikpy chain for xArm created successfully, including link_tcp from URDF.")
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

    def get_current_tcp_pose(self):
        """Gets the current TCP pose from /tf and returns it in dict format."""
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/link_base', '/link_tcp', rospy.Time(0))
            pose = {
                'position': {'x': trans[0], 'y': trans[1], 'z': trans[2]},
                'orientation': {'x': rot[0], 'y': rot[1], 'z': rot[2], 'w': rot[3]}
            }
            return pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF lookup failed when getting current TCP pose: {e}")
            return None

    def save_poses_to_json(self, filepath, poses):
        """
        Saves a list of poses to a JSON file.
        """
        rospy.loginfo(f"Saving {len(poses)} recorded xArm poses to {filepath}")
        try:
            with open(filepath, 'w') as f:
                json.dump(poses, f, indent=4)
            rospy.loginfo("Successfully saved xArm poses.")
        except IOError as e:
            rospy.logerr(f"Failed to write poses to {filepath}: {e}")

    def load_poses_as_matrices(self, filepath):
        """
        Loads poses and converts them directly to transformation matrices.
        """
        try:
            with open(filepath, 'r') as f:
                pose_list_raw = json.load(f)
            rospy.loginfo(f"Loaded {len(pose_list_raw)} raw poses from {filepath}")
        except IOError as e:
            rospy.logerr(f"Failed to read file {filepath}: {e}")
            return []

        target_matrices = []
        for p_raw in pose_list_raw:
            pos = p_raw['position']
            orient = p_raw['orientation']
            loaded_pose_matrix = tf_trans.quaternion_matrix([orient['x'], orient['y'], orient['z'], orient['w']])
            loaded_pose_matrix[0:3, 3] = [pos['x'], pos['y'], pos['z']]
            target_matrices.append(loaded_pose_matrix)
        return target_matrices

    def move_to_single_pose(self, target_matrix):
        """
        Calculates IK for a single pose, executes it, and records the final pose after completion.
        """
        rospy.loginfo("--- Moving to a single target pose ---")
        
        # Use the current robot position as the seed
        ik_seed = np.zeros(len(self.xarm_chain.links))
        ik_seed[self.xarm_chain.active_links_mask] = self.joint_positions

        # Calculate IK for the target pose
        target_joint_angles = self.xarm_chain.inverse_kinematics_frame(
            target_matrix,
            initial_position=ik_seed,
            orientation_mode="all"
        )

        if target_joint_angles is None:
            rospy.logerr("IK failed for the target pose. Aborting.")
            return

        # Create a trajectory with a single point
        ros_joint_angles = target_joint_angles[self.xarm_chain.active_links_mask]
        point = JointTrajectoryPoint(
            positions=ros_joint_angles.tolist(),
            time_from_start=rospy.Duration.from_sec(5.0) # Give it 5 seconds to get there
        )
        traj_msg = JointTrajectory(joint_names=self.joint_names, points=[point])
        goal_msg = FollowJointTrajectoryGoal(trajectory=traj_msg)

        rospy.loginfo("Sending single point goal to the action server...")
        self.action_client.send_goal(goal_msg)

        # Wait for the trajectory to finish
        if self.action_client.wait_for_result(rospy.Duration(10.0)):
            rospy.loginfo("Movement finished successfully.")
            
            # After movement is complete, get the final pose
            rospy.sleep(0.5) # Short pause to ensure TF buffer is updated
            final_pose = self.get_current_tcp_pose()

            if final_pose:
                self.recorded_xarm_poses = [final_pose] # Store only the single final pose
                output_filepath = "/home/gunjan/catkin_ws/src/franka_ros_code/franka_ros/Cross_Movement/xarm_Following/xarm_single_pose.json"
                self.save_poses_to_json(output_filepath, self.recorded_xarm_poses)
            else:
                rospy.logerr("Could not retrieve final TCP pose after movement.")

        else:
            rospy.logwarn("Movement timed out or was preempted.")

    

    def execute_ik_trajectory(self, target_matrices):
        """
        Calculates a joint trajectory from a list of Cartesian poses using ikpy and executes it.
        """
        rospy.loginfo("--- Executing full IK trajectory ---")
        
        trajectory_points = []
        time_from_start = 0.0
        segment_duration = 0.1 # seconds per segment

        # Use the current robot position as the seed for the first IK calculation
        # For subsequent IK calculations, use the previous IK result as the seed
        current_ik_seed = np.zeros(len(self.xarm_chain.links))
        if self.joint_positions is not None:
            current_ik_seed[self.xarm_chain.active_links_mask] = self.joint_positions
        else:
            rospy.logwarn("Initial joint positions not available, using zero seed for first IK.")

        for i, target_matrix in enumerate(target_matrices):
            
            target_joint_angles = self.xarm_chain.inverse_kinematics_frame(
                target_matrix,
                initial_position=current_ik_seed,
                orientation_mode="all"
            )

            if target_joint_angles is None:
                rospy.logerr(f"IK failed for target pose {i}. Skipping this point.")
                continue # Skip this point and try the next one

            ros_joint_angles = target_joint_angles[self.xarm_chain.active_links_mask]
            
            time_from_start += segment_duration
            point = JointTrajectoryPoint(
                positions=ros_joint_angles.tolist(),
                time_from_start=rospy.Duration.from_sec(time_from_start)
            )
            trajectory_points.append(point)
            
            # Update the seed for the next IK calculation
            current_ik_seed = target_joint_angles

        if not trajectory_points:
            rospy.logwarn("No valid trajectory points generated. Aborting playback.")
            return

        traj_msg = JointTrajectory(joint_names=self.joint_names, points=trajectory_points)
        goal_msg = FollowJointTrajectoryGoal(trajectory=traj_msg)

        rospy.loginfo(f"Sending trajectory with {len(trajectory_points)} points to the action server...")
        self.action_client.send_goal(goal_msg)

        # Wait for the trajectory to finish indefinitely
        if self.action_client.wait_for_result(rospy.Duration(0.0)):
            rospy.loginfo("Full trajectory playback finished successfully.")
        else:
            rospy.logwarn("Full trajectory playback timed out or was preempted.")

def main():
    try:
        player = XArmIKPlayer()

        poses_filepath = "/home/gunjan/catkin_ws/src/franka_ros_code/franka_ros/Cross_Movement/Franka_Recording/franka_poses.json"
        target_matrices = player.load_poses_as_matrices(poses_filepath)

        if not target_matrices:
            rospy.logerr("Aborting due to empty pose list.")
            return

        # --- EXPERIMENT: Move to the first pose only ---
        # first_pose_matrix = target_matrices[0]
        # player.move_to_single_pose(first_pose_matrix)

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
