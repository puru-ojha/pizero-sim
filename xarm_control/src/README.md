# xArm Control with Franka Policy

## Overview

This project demonstrates how to control a UFACTORY xArm 7 robot in a Gazebo simulation using a control policy originally developed for a Franka Emika Panda robot.

The core idea is to translate the observations from the xArm environment into a format that the Franka-based policy understands, and then translate the policy's actions back into commands for the xArm. This is achieved through kinematic transformations (Forward and Inverse Kinematics) between the two robot models.

## System Architecture

The system consists of three main components:

1.  **Gazebo Simulation**: An environment that simulates the xArm 7 robot, a table, a bowl, a marker pen, a wrist-mounted RealSense camera, and an external camera. This is launched via `xarm7_new.launch`.
2.  **Control Node (`camera_joint_control_node.py`)**: This Python script acts as a bridge. It:
    *   Subscribes to camera images and xArm joint states from Gazebo.
    *   Computes the equivalent Franka joint state from the xArm's current state.
    *   Sends the camera images and the emulated Franka joint state to a remote policy server.
    *   Receives action commands (joint velocities for a Franka) from the policy.
    *   Converts these actions into target joint positions for the xArm.
    *   Sends the commands to the xArm's joint trajectory controller.
3.  **Policy Server**: An external server (not included in this repository) that hosts the pre-trained control policy. It receives observations and returns actions over a websocket connection.

## Prerequisites

Before running the project, ensure you have the following dependencies installed and configured:

*   **ROS**: A working ROS installation (e.g., Melodic, Noetic).
*   **Catkin Workspace**: A configured catkin workspace where this project is located.
*   **Python Dependencies**:
    ```bash
    pip install numpy scipy
    ```
*   **`openpi-client`**: The required `openpi-client` library is included in the `src/` directory.
*   **Gazebo Models**: The simulation requires models for a `bowl` and a `marker_pen`. Ensure these models are available in your Gazebo models path (e.g., `~/.gazebo/models/`). The launch file references them directly.
*   **Policy Server**: The policy server must be running and accessible from the machine running the ROS nodes. The control node is hardcoded to connect to `ws://10.4.25.44:8000`.

## Installation and Setup

1.  **Build the Workspace**: From the root of your catkin workspace, run:
    ```bash
    catkin_make
    ```
2.  **Source the Workspace**: In every new terminal, source the setup file:
    ```bash
    source devel/setup.bash
    ```

## Running the Simulation

Follow these steps to run the full system.

### 1. Start the Policy Server

Ensure your external policy server is running and reachable at `ws://10.4.25.44:8000`. If your server is at a different address, you must update the IP address in `src/xarm_control/src/camera_joint_control_node.py`.

### 2. Launch the Gazebo Simulation

Open a new terminal, source your workspace, and run the following command to start Gazebo with the xArm robot and the environment:

```bash
roslaunch xarm_gazebo xarm7_new.launch
```

Gazebo will start in a paused state. The robot will be spawned on a table with a bowl and a marker pen nearby.

### 3. Run the Control Node

Open another terminal, source your workspace, and run the control node:

```bash
rosrun xarm_control camera_joint_control_node.py
```

This node will:
1.  Connect to the policy server.
2.  Move the xArm to a default starting position.
3.  Begin the main control loop, sending observations to the policy and executing the returned actions.

## How It Works: The Kinematic Bridge

This project enables a control policy designed for a Franka Emika Panda robot to control an xArm 7 robot by bridging the kinematic differences between them. This involves translating observations from the xArm to a Franka-compatible format and converting Franka-based actions back into xArm commands.

### Workflow in Simple Terms

1.  **Robot Initialization**:
    *   The xArm first moves to its home (all zeros) position.
    *   Then, it slowly moves to a predefined policy start position.
    *   During this gradual movement, the system continuously calculates the Franka robot's equivalent joint configuration that matches the xArm's current pose. This ensures a highly accurate starting point for the Franka's joint states, which is crucial for the policy.

2.  **Observation Processing (xArm -> Franka)**:
    *   The control node captures real-time camera images and reads the xArm's current joint states.
    *   It uses Forward Kinematics (FK) to determine the 3D end-effector pose of the xArm.
    *   It then uses Inverse Kinematics (IK) to find the Franka robot's joint angles that would achieve this exact same end-effector pose.
    *   These "emulated" Franka joint angles, along with the camera images, are sent as an "observation" to the remote policy server.

3.  **Action Execution (Franka -> xArm)**:
    *   The policy server, based on the received observation, computes and returns an "action" (typically a set of joint velocity commands for the Franka).
    *   The control node integrates these Franka velocities over a small time step to determine a target Franka joint configuration.
    *   It then uses FK to find the 3D end-effector pose corresponding to this target Franka configuration.
    *   Finally, it uses IK to calculate the necessary xArm joint angles to reach that target end-effector pose.
    *   These calculated xArm joint angles are sent as a trajectory goal to the xArm's joint controller, causing the robot to move.

This continuous loop of observation, policy inference, and action execution allows the Franka-specific policy to effectively control the xArm robot in the simulation.

### URDFs for ikpy

To ensure compatibility and accuracy with the `ikpy` library for kinematic calculations, new URDF (Unified Robot Description Format) files have been created for both the Franka (`panda_nomesh.urdf`) and xArm (`xarm7_nomesh.urdf`). These URDFs are specifically designed to work seamlessly with `ikpy`, addressing limitations of previous URDF versions that were not directly suitable for `ikpy`'s requirements.

## Key Files

*   `src/xarm_control/src/camera_joint_control_node.py`: The core control logic, including ROS communication, policy interaction, and kinematic conversions.
*   `src/xarm_ros_code/xarm_ros/xarm_gazebo/launch/xarm7_new.launch`: The main launch file that sets up the entire Gazebo simulation environment.
*   `src/xarm_ros_code/xarm_ros/xarm_description/urdf/xarm_device.urdf.xacro`: The URDF description for the xArm robot.
