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
        Loads poses from the JSON file and converts them into a list of 4x4 transformation matrices
        that ikpy can use. No coordinate system transformation is done here, assuming the
        poses in the file are already in the xArm's base frame.
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
            
            # Apply a fixed rotation to account for different end-effector frame conventions
            # Franka EEF: X=left, Y=up, Z=forward
            # xArm EEF: X=forward, Y=left, Z=up
            # # Rotation: -90 deg around Y, then -90 deg around new X
            # q1 = tf_trans.quaternion_about_axis(-np.pi/2, [0, 1, 0]) # -90 deg around Y
            # q2 = tf_trans.quaternion_about_axis(-np.pi/2, [1, 0, 0]) # -90 deg around X
            # correction_q = tf_trans.quaternion_multiply(q2, q1)
            
            # Multiply the recorded orientation by the correction quaternion
            # Order matters: correction_q * recorded_q (applying correction before original)
            recorded_q = [orient['x'], orient['y'], orient['z'], orient['w']]
            # corrected_q = tf_trans.quaternion_multiply(correction_q, recorded_q)
            corrected_q = recorded_q


            # Create a 4x4 transformation matrix from position and the corrected quaternion
            matrix = tf_trans.quaternion_matrix(corrected_q)
            matrix[0, 3] = pos['x']
            matrix[1, 3] = pos['y']
            matrix[2, 3] = pos['z']
            target_matrices.append(matrix)
            
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
