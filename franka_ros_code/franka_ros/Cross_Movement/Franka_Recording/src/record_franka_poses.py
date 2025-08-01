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

    def record_plan_to_poses(self, plan):
        """
        Converts a joint-space plan into a list of Cartesian poses using Forward Kinematics.
        """
        if not plan.joint_trajectory.points:
            rospy.logerr("Input plan contains no points. Cannot record poses.")
            return []

        recorded_poses = []
        joint_names = plan.joint_trajectory.joint_names
        header = plan.joint_trajectory.header

        rospy.loginfo("Performing Forward Kinematics for each point in the planned trajectory...")
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
        
        rospy.loginfo("Converted plan to {} Cartesian poses.".format(len(recorded_poses)))
        return recorded_poses

    def plan_and_execute_to_pose(self, target_pose, all_poses_list):
        """
        Plans a motion to a target pose, records the plan, and then executes it.
        """
        self.arm_move_group.set_pose_target(target_pose)
        
        # Plan the motion - this returns a tuple in some MoveIt versions
        plan_result = self.arm_move_group.plan()

        # Extract the actual plan from the tuple
        plan = plan_result[1]  # The RobotTrajectory is the second element

        if not plan.joint_trajectory.points:
            rospy.logerr("Could not create a plan to the target pose. Aborting segment.")
            self.arm_move_group.clear_pose_targets()
            return False

        # Record the planned path before execution
        rospy.loginfo("Recording planned trajectory...")
        segment_poses = self.record_plan_to_poses(plan)
        if segment_poses:
            all_poses_list.extend(segment_poses)
        
        # Execute the planned path
        rospy.loginfo("Executing planned trajectory...")
        success = self.arm_move_group.execute(plan, wait=True)
        
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        
        if not success:
            rospy.logwarn("Failed to execute the planned trajectory.")

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
        z_offset = 0 # 0 cm

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
        waypoints = {
            "home_pose": home_pose,
            "marker_pre_grasp": marker_pre_grasp,
            "marker_grasp": marker_grasp,
            "marker_post_grasp": marker_post_grasp,
            "bowl_pre_drop": bowl_pre_drop,
            "bowl_drop": bowl_drop,
            "bowl_post_drop": bowl_post_drop,
            "home_pose_return": home_pose
        }

        # --- 2. Plan, Record, and Execute the Full Trajectory ---
        rospy.loginfo("--- Starting Trajectory Planning, Recording, and Execution Phase ---")
        full_trajectory = []

        # Move to initial home pose
        rospy.loginfo("Moving to home pose.")
        recorder.plan_and_execute_to_pose(waypoints["home_pose"], full_trajectory)
        
        # Execute pick sequence
        rospy.loginfo("Executing pick sequence.")
        recorder.operate_gripper("open")
        recorder.plan_and_execute_to_pose(waypoints["marker_pre_grasp"], full_trajectory)
        recorder.plan_and_execute_to_pose(waypoints["marker_grasp"], full_trajectory)
        recorder.operate_gripper("close")
        recorder.plan_and_execute_to_pose(waypoints["marker_post_grasp"], full_trajectory)

        # Execute place sequence
        rospy.loginfo("Executing place sequence.")
        recorder.plan_and_execute_to_pose(waypoints["bowl_pre_drop"], full_trajectory)
        recorder.plan_and_execute_to_pose(waypoints["bowl_drop"], full_trajectory)
        recorder.operate_gripper("open")
        recorder.plan_and_execute_to_pose(waypoints["bowl_post_drop"], full_trajectory)

        # Return to home pose
        rospy.loginfo("Returning to home pose.")
        recorder.plan_and_execute_to_pose(waypoints["home_pose_return"], full_trajectory)

        # --- 3. Save the Recorded Trajectory ---
        if not full_trajectory:
            rospy.logerr("Recording failed, no poses were captured. Aborting save.")
            return

        recorder.save_poses_to_file(full_trajectory)

        rospy.loginfo("Pick and place routine finished.")

    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()