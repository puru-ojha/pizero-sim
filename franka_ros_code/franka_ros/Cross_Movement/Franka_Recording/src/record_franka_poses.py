#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from geometry_msgs.msg import Pose, Point, Quaternion
import json
import os

class FrankaRecorder:
    def __init__(self):
        """
        Initializes the FrankaRecorder, connecting to MoveIt for both the arm and the hand.
        """
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('franka_recorder', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Setup move group for the arm
        self.arm_group_name = "panda_arm"
        self.arm_move_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        self.eef_link = self.arm_move_group.get_end_effector_link()
        rospy.loginfo("Using end-effector link: {}".format(self.eef_link))

        # Setup move group for the gripper
        self.hand_group_name = "panda_hand"
        self.hand_move_group = moveit_commander.MoveGroupCommander(self.hand_group_name)

        # Connect to the MoveIt FK service
        rospy.loginfo("Waiting for '/compute_fk' service...")
        rospy.wait_for_service('compute_fk')
        self.fk_service = rospy.ServiceProxy('compute_fk', GetPositionFK)
        rospy.loginfo("Service '/compute_fk' connected.")

    def operate_gripper(self, action):
        """
        Controls the gripper.
        Args:
            action (str): "open" or "close".
        """
        joint_goal = self.hand_move_group.get_current_joint_values()
        if action == "open":
            rospy.loginfo("Opening gripper.")
            joint_goal[0] = 0.04
            joint_goal[1] = 0.04
        elif action == "close":
            rospy.loginfo("Closing gripper.")
            joint_goal[0] = 0.0
            joint_goal[1] = 0.0
        else:
            rospy.logwarn("Invalid gripper action specified.")
            return False

        self.hand_move_group.go(joint_goal, wait=True)
        self.hand_move_group.stop()
        return True

    def record_trajectory_from_waypoints(self, waypoints):
        """
        Computes a Cartesian path through a list of waypoints and records the full trajectory.
        Does NOT execute the plan.
        """
        if not waypoints:
            rospy.logerr("Waypoint list is empty. Cannot record trajectory.")
            return []

        rospy.loginfo("Computing Cartesian path through {} waypoints...".format(len(waypoints)))
        # Compute the Cartesian path. eef_step is the resolution.
        # The third argument, jump_threshold, is set to False to disable it, matching the C++ signature.
        (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            waypoints, 0.01, False
        )

        rospy.loginfo("Cartesian path computed. Fraction of path achieved: {:.2f}%".format(fraction * 100))

        if fraction < 0.9: # Allow for some tolerance
            rospy.logwarn("Could not compute a complete Cartesian path. The recorded trajectory may be incomplete.")

        if not plan.joint_trajectory.points:
            rospy.logerr("Trajectory planning failed. The plan contains no points.")
            return []

        recorded_poses = []
        joint_names = plan.joint_trajectory.joint_names
        header = plan.joint_trajectory.header

        rospy.loginfo("Performing Forward Kinematics for each point in the trajectory...")
        for point in plan.joint_trajectory.points:
            fk_request = GetPositionFKRequest()
            fk_request.header = header
            fk_request.fk_link_names = [self.eef_link]
            fk_request.robot_state.joint_state.name = joint_names
            fk_request.robot_state.joint_state.position = point.positions
            try:
                fk_response = self.fk_service(fk_request)
                if fk_response.error_code.val == fk_response.error_code.SUCCESS:
                    recorded_poses.append(fk_response.pose_stamped[0].pose)
                else:
                    rospy.logerr("FK failed with error code: {}".format(fk_response.error_code.val))
            except rospy.ServiceException as e:
                rospy.logerr("FK service call failed: {}".format(e))
        
        rospy.loginfo("Recorded {} total poses for the trajectory.".format(len(recorded_poses)))
        return recorded_poses

    def execute_move_to_pose(self, target_pose):
        """
        Plans and executes a move to a target pose.
        """
        self.arm_move_group.set_pose_target(target_pose)
        success = self.arm_move_group.go(wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        return success

    def save_poses_to_file(self, poses, filename="franka_poses.json"):
        """
        Saves a list of Pose objects to a JSON file.
        """
        pose_list = []
        for pose in poses:
            p_dict = {
                'position': {'x': pose.position.x, 'y': pose.position.y, 'z': pose.position.z},
                'orientation': {'x': pose.orientation.x, 'y': pose.orientation.y, 'z': pose.orientation.z, 'w': pose.orientation.w}
            }
            pose_list.append(p_dict)

        output_dir = "/home/gunjan/catkin_ws/src/franka_ros_code/franka_ros/Cross_Movement/Franka_Recording"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(pose_list, f, indent=4)
            rospy.loginfo("Successfully saved {} total poses to {}".format(len(pose_list), filepath))
        except IOError as e:
            rospy.logerr("Failed to write to {}: {}".format(filepath, e))

def main():
    try:
        recorder = FrankaRecorder()

        # --- Define Poses in Robot's Base Frame ---
        # VERTICAL OFFSET to avoid collision due to gripper length mismatch
        z_offset = 0.15 # 15 cm

        # Orientation for a top-down grasp
        grasp_orientation = {'x': -1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}

        home_pose = Pose(
            position=Point(x=0.3, y=0.0, z=0.5 + z_offset),
            orientation=Quaternion(x=grasp_orientation['x'], y=grasp_orientation['y'], z=grasp_orientation['z'], w=grasp_orientation['w'])
        )

        # --- Marker Pick Sequence ---
        marker_pre_grasp = Pose(
            position=Point(x=0.4, y=-0.2, z=0.25 + z_offset),
            orientation=Quaternion(x=grasp_orientation['x'], y=grasp_orientation['y'], z=grasp_orientation['z'], w=grasp_orientation['w'])
        )
        marker_grasp = Pose(
            position=Point(x=0.4, y=-0.2, z=0.12 + z_offset),
            orientation=Quaternion(x=grasp_orientation['x'], y=grasp_orientation['y'], z=grasp_orientation['z'], w=grasp_orientation['w'])
        ) # Z adjusted for grasp
        marker_post_grasp = Pose(
            position=Point(x=0.4, y=-0.2, z=0.25 + z_offset),
            orientation=Quaternion(x=grasp_orientation['x'], y=grasp_orientation['y'], z=grasp_orientation['z'], w=grasp_orientation['w'])
        )

        # --- Bowl Place Sequence ---
        bowl_pre_drop = Pose(
            position=Point(x=0.4, y=0.2, z=0.25 + z_offset),
            orientation=Quaternion(x=grasp_orientation['x'], y=grasp_orientation['y'], z=grasp_orientation['z'], w=grasp_orientation['w'])
        )
        bowl_drop = Pose(
            position=Point(x=0.4, y=0.2, z=0.15 + z_offset),
            orientation=Quaternion(x=grasp_orientation['x'], y=grasp_orientation['y'], z=grasp_orientation['z'], w=grasp_orientation['w'])
        )
        bowl_post_drop = Pose(
            position=Point(x=0.4, y=0.2, z=0.25 + z_offset),
            orientation=Quaternion(x=grasp_orientation['x'], y=grasp_orientation['y'], z=grasp_orientation['z'], w=grasp_orientation['w'])
        )

        # --- 1. Define Waypoints for the Full Trajectory ---
        waypoints = []
        waypoints.append(home_pose)
        waypoints.append(marker_pre_grasp)
        waypoints.append(marker_grasp)
        waypoints.append(marker_post_grasp)
        waypoints.append(bowl_pre_drop)
        waypoints.append(bowl_drop)
        waypoints.append(bowl_post_drop)
        waypoints.append(home_pose)

        # --- 2. Record Full Trajectory without Executing ---
        rospy.loginfo("--- Starting Trajectory Recording Phase ---")
        full_trajectory = recorder.record_trajectory_from_waypoints(waypoints)

        if not full_trajectory:
            rospy.logerr("Recording failed, no poses were captured. Aborting.")
            return

        recorder.save_poses_to_file(full_trajectory)

        # --- 3. Execute the Full Pick-and-Place Motion for Confirmation ---
        rospy.loginfo("--- Starting Motion Execution Phase ---")
        rospy.loginfo("Moving to home pose.")
        recorder.execute_move_to_pose(home_pose)
        
        rospy.loginfo("Executing pick sequence.")
        recorder.operate_gripper("open")
        recorder.execute_move_to_pose(marker_pre_grasp)
        recorder.execute_move_to_pose(marker_grasp)
        recorder.operate_gripper("close")
        recorder.execute_move_to_pose(marker_post_grasp)

        rospy.loginfo("Executing place sequence.")
        recorder.execute_move_to_pose(bowl_pre_drop)
        recorder.execute_move_to_pose(bowl_drop)
        recorder.operate_gripper("open")
        recorder.execute_move_to_pose(bowl_post_drop)

        rospy.loginfo("Returning to home pose.")
        recorder.execute_move_to_pose(home_pose)

        rospy.loginfo("Pick and place routine finished.")

    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()