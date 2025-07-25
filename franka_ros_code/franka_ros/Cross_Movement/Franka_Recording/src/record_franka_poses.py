#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from geometry_msgs.msg import Pose
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

    def record_plan_to_pose(self, target_pose):
        """
        Plans a trajectory and uses FK to record the Cartesian path.
        Does NOT execute the plan.
        """
        self.arm_move_group.set_pose_target(target_pose)
        plan_success, plan, _, _ = self.arm_move_group.plan()

        if not plan_success:
            rospy.logerr("Trajectory planning failed.")
            return []

        recorded_poses = []
        joint_names = plan.joint_trajectory.joint_names
        header = plan.joint_trajectory.header

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
                    rospy.logerr("FK failed with error: {}".format(fk_response.error_code.val))
            except rospy.ServiceException as e:
                rospy.logerr("FK service call failed: {}".format(e))
        
        rospy.loginfo(f"Recorded {len(recorded_poses)} poses for the segment.")
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
            rospy.loginfo(f"Successfully saved {len(pose_list)} total poses to {filepath}")
        except IOError as e:
            rospy.logerr(f"Failed to write to {filepath}: {e}")

def main():
    try:
        recorder = FrankaRecorder()
        full_trajectory = []

        # --- Define Poses in Robot's Base Frame ---
        # Coordinates are derived from Gazebo world poses minus the robot's spawn pose.
        # Robot base is at x=-0.2, y=-0.8, z=1.021
        # Marker is at   x=0.2,  y=-1.0, z=1.021 -> Robot frame: (0.4, -0.2, 0.0)
        # Bowl is at     x=0.2,  y=-0.6, z=1.021 -> Robot frame: (0.4,  0.2, 0.0)

        # Orientation for a top-down grasp
        grasp_orientation = {'x': -1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}

        home_pose = Pose()
        home_pose.position.x = 0.3
        home_pose.position.y = 0.0
        home_pose.position.z = 0.5
        home_pose.orientation.x = grasp_orientation['x']
        home_pose.orientation.y = grasp_orientation['y']
        home_pose.orientation.z = grasp_orientation['z']
        home_pose.orientation.w = grasp_orientation['w']

        # --- Marker Pick Sequence ---
        marker_pre_grasp = Pose(position={'x': 0.4, 'y': -0.2, 'z': 0.25}, orientation=grasp_orientation)
        marker_grasp = Pose(position={'x': 0.4, 'y': -0.2, 'z': 0.12}, orientation=grasp_orientation) # Z adjusted for grasp
        marker_post_grasp = Pose(position={'x': 0.4, 'y': -0.2, 'z': 0.25}, orientation=grasp_orientation)

        # --- Bowl Place Sequence ---
        bowl_pre_drop = Pose(position={'x': 0.4, 'y': 0.2, 'z': 0.25}, orientation=grasp_orientation)
        bowl_drop = Pose(position={'x': 0.4, 'y': 0.2, 'z': 0.15}, orientation=grasp_orientation)
        bowl_post_drop = Pose(position={'x': 0.4, 'y': 0.2, 'z': 0.25}, orientation=grasp_orientation)

        # --- 1. Record Full Trajectory without Executing ---
        rospy.loginfo("--- Starting Trajectory Recording Phase ---")
        # Move from home to pre-grasp
        full_trajectory.extend(recorder.record_plan_to_pose(marker_pre_grasp))
        # Move to grasp
        full_trajectory.extend(recorder.record_plan_to_pose(marker_grasp))
        # Lift marker
        full_trajectory.extend(recorder.record_plan_to_pose(marker_post_grasp))
        # Move to pre-drop over bowl
        full_trajectory.extend(recorder.record_plan_to_pose(bowl_pre_drop))
        # Lower to drop
        full_trajectory.extend(recorder.record_plan_to_pose(bowl_drop))
        # Retract from bowl
        full_trajectory.extend(recorder.record_plan_to_pose(bowl_post_drop))
        # Return home
        full_trajectory.extend(recorder.record_plan_to_pose(home_pose))

        if not full_trajectory:
            rospy.logerr("Recording failed, no poses were captured. Aborting.")
            return

        recorder.save_poses_to_file(full_trajectory)

        # --- 2. Execute the Full Pick-and-Place Motion for Confirmation ---
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