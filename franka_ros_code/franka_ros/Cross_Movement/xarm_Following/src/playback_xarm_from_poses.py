#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import json
import os
import tf
from tf.transformations import quaternion_from_euler, quaternion_multiply

class XArmPlayer:
    def __init__(self):
        """
        Initializes the XArmPlayer node, connecting to MoveIt for the arm and gripper.
        """
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('xarm_player', anonymous=True)

        # Setup move group for the xArm arm
        self.arm_group_name = "xarm7"
        self.arm_move_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        self.arm_move_group.set_planning_time(30) # Increase planning time for complex paths

        # Setup move group for the xArm gripper
        self.gripper_group_name = "xarm_gripper"
        self.gripper_move_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)

        rospy.loginfo("xArm Player node initialized.")

    def load_and_transform_poses(self, filepath):
        """
        Loads poses from a JSON file and transforms them from the Franka's coordinate
        system to the xArm's coordinate system.
        """
        try:
            with open(filepath, 'r') as f:
                pose_list_raw = json.load(f)
        except IOError as e:
            rospy.logerr(f"Failed to read file {filepath}: {e}")
            return []

        transformed_poses = []
        # Transformation from Franka base to xArm base:
        # 1. Translate by (0, -0.2, 0)
        # 2. Rotate by -90 degrees ( -pi/2 radians) around Z-axis
        q_rotation = quaternion_from_euler(0, 0, -1.570796)

        for p_raw in pose_list_raw:
            p_franka = p_raw['position']
            o_franka = p_raw['orientation']

            # Apply translation
            p_translated = {
                'x': p_franka['x'],
                'y': p_franka['y'] - 0.2,
                'z': p_franka['z']
            }

            # Apply rotation to the position vector
            p_rotated_x = p_translated['y'] # cos(-90)*x - sin(-90)*y = y
            p_rotated_y = -p_translated['x'] # sin(-90)*x + cos(-90)*y = -x

            # Apply rotation to the orientation quaternion
            q_franka = [o_franka['x'], o_franka['y'], o_franka['z'], o_franka['w']]
            q_rotated = quaternion_multiply(q_rotation, q_franka)

            pose_goal = Pose()
            pose_goal.position.x = p_rotated_x
            pose_goal.position.y = p_rotated_y
            pose_goal.position.z = p_franka['z'] # Z is unaffected
            pose_goal.orientation.x = q_rotated[0]
            pose_goal.orientation.y = q_rotated[1]
            pose_goal.orientation.z = q_rotated[2]
            pose_goal.orientation.w = q_rotated[3]
            transformed_poses.append(pose_goal)
        
        rospy.loginfo(f"Loaded and transformed {len(transformed_poses)} poses.")
        return transformed_poses

    def move_to_pose(self, target_pose):
        """
        Plans and executes a move to a single target pose.
        """
        self.arm_move_group.set_pose_target(target_pose)
        self.arm_move_group.set_start_state_to_current_state()
        success = self.arm_move_group.go(wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        return success

    def execute_cartesian_path(self, waypoints):
        """
        Computes and executes a Cartesian path through the given waypoints.
        """
        if not waypoints:
            rospy.logerr("Waypoint list is empty. Cannot execute path.")
            return

        (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                                   waypoints,   # list of poses
                                   0.01,        # eef_step
                                   0.0)         # jump_threshold

        rospy.loginfo(f"Cartesian path computed. Fraction of path planned: {fraction:.2f}")

        if fraction < 0.9:
            rospy.logerr("Could not compute the full Cartesian path. Aborting execution.")
            return

        rospy.loginfo("Executing the planned path.")
        self.arm_move_group.execute(plan, wait=True)

    def operate_gripper(self, action):
        """
        Controls the xArm gripper.
        Args:
            action (str): "open" or "close".
        """
        joint_goal = self.gripper_move_group.get_current_joint_values()
        if action == "open":
            rospy.loginfo("Opening gripper.")
            joint_goal = [0.85] * len(joint_goal) # 0.85 is fully open for xArm
        elif action == "close":
            rospy.loginfo("Closing gripper.")
            joint_goal = [0.0] * len(joint_goal) # 0 is closed
        else:
            rospy.logwarn("Invalid gripper action specified.")
            return False

        self.gripper_move_group.go(joint_goal, wait=True)
        self.gripper_move_group.stop()
        return True

def main():
    try:
        player = XArmPlayer()

        # Path to the recorded poses file
        poses_filepath = "/home/gunjan/catkin_ws/src/franka_ros_code/franka_ros/Cross_Movement/Franka_Recording/franka_poses.json"

        # Load and transform the poses
        waypoints = player.load_and_transform_poses(poses_filepath)

        if not waypoints:
            rospy.logerr("Aborting due to empty waypoint list.")
            return

        # --- Execute the full sequence ---
        rospy.loginfo("--- Starting xArm Playback ---")
        
        # 1. Move to the starting pose of the trajectory
        start_pose = waypoints[0]
        rospy.loginfo("Moving to the initial trajectory pose...")
        player.arm_move_group.set_pose_target(start_pose)
        if not player.arm_move_group.go(wait=True):
            rospy.logerr("Failed to move to the starting pose. Aborting.")
            return
        player.arm_move_group.stop()
        player.arm_move_group.clear_pose_targets()
        rospy.loginfo("Reached starting pose.")

        # 2. Execute the main trajectory
        player.operate_gripper("open")
        rospy.sleep(1)

        rospy.loginfo("Executing the full recorded trajectory.")
        player.execute_cartesian_path(waypoints[1:])
        rospy.sleep(1)

        # 3. Final gripper actions (simplified)
        player.operate_gripper("close")
        rospy.sleep(1)
        player.operate_gripper("open")

        rospy.loginfo("Playback finished.")

    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()