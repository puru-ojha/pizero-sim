#!/usr/bin/env python3
import rospy
import numpy as np
import ikpy.chain
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from actionlib import SimpleActionClient
import moveit_commander
import sys
import copy
from scipy.spatial.transform import Rotation as R

# A canonical default "ready" pose for the Franka robot
FRANKA_CANONICAL_READY_POSE = [0, -0.785, 0, -2.356, 0, 1.57, 0.785]
XARM_INITIAL_POSE = [-2.5, -0.6, 0.5, 1.2, 0.0, 1.2, 0.0]

class FrankaXArmImitationNode:
    def __init__(self):
        rospy.init_node('franka_xarm_imitation_node', anonymous=True)
        rospy.loginfo("Franka to xArm Imitation Node Started")

        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        # --- Kinematics Setup (ikpy) ---
        rospy.loginfo("Loading robot models for ikpy...")
        try:
            self.franka_chain = ikpy.chain.Chain.from_urdf_file(
                "/home/gunjan/catkin_ws/panda_nomesh.urdf",
                active_links_mask=[False, True, True, True, True, True, True, True, False]
            )
            self.xarm_chain = ikpy.chain.Chain.from_urdf_file(
                "/home/gunjan/catkin_ws/xarm7_nomesh.urdf",
                base_elements=["link_base"],
                active_links_mask=[False, True, True, True, True, True, True, True, False]
            )
            rospy.loginfo("ikpy chains created successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load URDFs for ikpy: {e}")
            return

        # --- Action Client Setup ---
        self.action_client = SimpleActionClient(
            '/xarm/xarm7_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for xArm action server...")
        if not self.action_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("Failed to connect to xArm action server. Exiting.")
            return
        rospy.loginfo("xArm action server connected.")

        # --- Main Execution ---
        # Initialize MoveIt! components once
        rospy.loginfo("Initializing MoveIt!...")
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm") # Assuming "panda_arm" is the planning group

        self.run_imitation()

    def generate_franka_trajectory_with_moveit(self):
        rospy.loginfo("Generating Franka pick-and-place trajectory with MoveIt!...")

        # Gripper pointing down orientation
        orientation_down = R.from_euler('x', 180, degrees=True).as_quat()

        # Define waypoints for the pick and place task
        pen_pos = [0.5, 0, 0.1]
        bowl_pos = [0, 0.5, 0.1]

        waypoints = []
        # Start
        wpose = self.group.get_current_pose().pose
        wpose.position.x = 0.3
        wpose.position.y = 0
        wpose.position.z = 0.5
        wpose.orientation.x = orientation_down[0]
        wpose.orientation.y = orientation_down[1]
        wpose.orientation.z = orientation_down[2]
        wpose.orientation.w = orientation_down[3]
        waypoints.append(copy.deepcopy(wpose))

        # Pre-pick
        wpose.position.x = pen_pos[0]
        wpose.position.y = pen_pos[1]
        wpose.position.z = pen_pos[2] + 0.1
        waypoints.append(copy.deepcopy(wpose))

        # Pick
        wpose.position.z = pen_pos[2]
        waypoints.append(copy.deepcopy(wpose))

        # Post-pick
        wpose.position.z = pen_pos[2] + 0.1
        waypoints.append(copy.deepcopy(wpose))

        # Pre-place
        wpose.position.x = bowl_pos[0]
        wpose.position.y = bowl_pos[1]
        wpose.position.z = bowl_pos[2] + 0.1
        waypoints.append(copy.deepcopy(wpose))

        # Place
        wpose.position.z = bowl_pos[2]
        waypoints.append(copy.deepcopy(wpose))

        # Post-place
        wpose.position.z = bowl_pos[2] + 0.1
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = self.group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   0.01,        # eef_step
                                   0.0)         # jump_threshold

        if fraction < 1.0:
            rospy.logwarn(f"Only {fraction * 100}% of the trajectory was planned. The robot may not be able to reach all waypoints.")

        rospy.loginfo(f"Generated Franka trajectory with {len(plan.joint_trajectory.points)} points.")
        return plan.joint_trajectory

    def send_trajectory_to_xarm(self, joint_positions):
        traj_msg = JointTrajectory(
            joint_names=self.joint_names,
            points=[JointTrajectoryPoint(positions=joint_positions, time_from_start=rospy.Duration(0.1))]
        )
        goal_msg = FollowJointTrajectoryGoal(trajectory=traj_msg)
        self.action_client.send_goal(goal_msg)
        if not self.action_client.wait_for_result(rospy.Duration(2.0)):
            rospy.logwarn("Trajectory execution timed out.")
            return False
        return self.action_client.get_result().error_code == FollowJointTrajectoryGoal.SUCCESSFUL

    def run_imitation(self):
        # 2. Move xArm to its initial pose
        rospy.loginfo("Moving xArm to initial pose...")
        if not self.send_trajectory_to_xarm(XARM_INITIAL_POSE):
            rospy.logerr("Failed to move xArm to initial pose. Aborting.")
            return
        rospy.loginfo("xArm at initial pose.")
        rospy.sleep(1)

        # 3. Synchronize Franka to xArm's starting pose
        rospy.loginfo("Synchronizing Franka to xArm's starting pose...")
        xarm_ik_start = np.zeros(len(self.xarm_chain.links))
        xarm_ik_start[self.xarm_chain.active_links_mask] = XARM_INITIAL_POSE
        xarm_start_pose_matrix = self.xarm_chain.forward_kinematics(xarm_ik_start)
        
        # Convert 4x4 matrix to PoseStamped for MoveIt!
        start_pose = self.group.get_current_pose()
        start_pose.pose.position.x = xarm_start_pose_matrix[0, 3]
        start_pose.pose.position.y = xarm_start_pose_matrix[1, 3]
        start_pose.pose.position.z = xarm_start_pose_matrix[2, 3]
        
        orientation_quat = R.from_matrix(xarm_start_pose_matrix[:3, :3]).as_quat()
        start_pose.pose.orientation.x = orientation_quat[0]
        start_pose.pose.orientation.y = orientation_quat[1]
        start_pose.pose.orientation.z = orientation_quat[2]
        start_pose.pose.orientation.w = orientation_quat[3]

        self.group.set_pose_target(start_pose)
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        if not plan:
            rospy.logerr("Failed to plan and move Franka to the synchronized start pose. Aborting.")
            return
        rospy.loginfo("Franka synchronized with xArm.")

        # 4. Generate the leader's (Franka) trajectory from the new starting position
        franka_trajectory = self.generate_franka_trajectory_with_moveit()

        # 5. Execute the imitation loop
        rospy.loginfo("Starting cross-robot imitation...")
        rate = rospy.Rate(10) # Control loop frequency
        
        # Seed the first xArm IK with its current known position
        xarm_ik_seed = np.zeros(len(self.xarm_chain.links))
        xarm_ik_seed[self.xarm_chain.active_links_mask] = XARM_INITIAL_POSE

        for i, franka_joint_point in enumerate(franka_trajectory.points):
            if rospy.is_shutdown():
                break
            
            rospy.loginfo_throttle(1, f"Executing step {i+1}/{len(franka_trajectory.points)}")

            # Convert franka joint positions to ikpy format
            franka_joints_ikpy = np.zeros(len(self.franka_chain.links))
            franka_joints_ikpy[self.franka_chain.active_links_mask] = franka_joint_point.positions

            # FK on Franka to get end-effector pose
            franka_pose = self.franka_chain.forward_kinematics(franka_joints_ikpy)

            # IK for xArm to find joint angles for that pose
            target_xarm_ikpy = self.xarm_chain.inverse_kinematics_frame(
                target=franka_pose,
                initial_position=xarm_ik_seed
            )

            if target_xarm_ikpy is not None:
                target_xarm_ros = target_xarm_ikpy[self.xarm_chain.active_links_mask]
                
                # Send command to xArm
                if not self.send_trajectory_to_xarm(target_xarm_ros):
                    rospy.logwarn("Failed to execute trajectory on xArm. Skipping step.")
                
                # Update the seed for the next IK calculation for smoother motion
                xarm_ik_seed = target_xarm_ikpy
            else:
                rospy.logwarn("IK solution for xArm not found. Skipping step.")
            
            rate.sleep()

        rospy.loginfo("Imitation task finished.")

if __name__ == '__main__':
    try:
        FrankaXArmImitationNode()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("Shutting down imitation node.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
