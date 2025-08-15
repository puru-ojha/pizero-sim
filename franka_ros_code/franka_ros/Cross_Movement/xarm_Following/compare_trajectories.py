import json
import matplotlib.pyplot as plt
import numpy as np
import tf.transformations as tf_trans
import os

def load_trajectory_data(filepath):
    """Loads position and orientation data from a JSON file.
    If 'time_from_start' is present, it's also returned.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    positions = []
    orientations_quat = [] # x, y, z, w
    times = [] # To store time_from_start if available
    
    for point in data:
        pos = point['position']
        orient = point['orientation']
        positions.append([pos['x'], pos['y'], pos['z']])
        orientations_quat.append([orient['x'], orient['y'], orient['z'], orient['w']])
        
        if 'time_from_start' in point:
            times.append(point['time_from_start'])
        
    return np.array(positions), np.array(orientations_quat), np.array(times) if times else None

def unwrap_quaternions(quaternions):
    """
    Processes an array of quaternions to ensure continuity in representation.
    If the dot product of consecutive quaternions is negative, one is negated.
    """
    unwrapped_q = [quaternions[0]]
    for i in range(1, len(quaternions)):
        q_prev = unwrapped_q[-1]
        q_curr = quaternions[i]
        dot_product = np.dot(q_prev, q_curr)
        if dot_product < 0:
            unwrapped_q.append(-q_curr) # Negate to make it continuous
        else:
            unwrapped_q.append(q_curr)
    return np.array(unwrapped_q)

def plot_trajectory_comparison(franka_data, xarm_data, title_prefix, y_label, index, franka_times, xarm_times):
    """
    Generates a plot comparing a specific coordinate/orientation over time.
    """
    
    # Use Franka's actual times
    franka_time = franka_times

    # Use xArm's actual recorded times
    xarm_time = xarm_times

    plt.figure(figsize=(10, 6))
    plt.plot(franka_time, franka_data[:, index], label='Franka', color='blue')
    plt.plot(xarm_time, xarm_data[:, index], label='xArm', color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel(y_label)
    plt.title(f'{title_prefix} Comparison over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{title_prefix.replace(" ", "_").lower()}_comparison.png')
    plt.close()

if __name__ == "__main__":
    franka_filepath = "/home/gunjan/catkin_ws/src/franka_ros_code/franka_ros/Cross_Movement/Franka_Recording/franka_poses.json"
    xarm_filepath = "/home/gunjan/catkin_ws/src/franka_ros_code/franka_ros/Cross_Movement/xarm_Following/xarm_actual_trajectory.json"

    # Load data
    franka_positions, franka_orientations_quat, franka_times = load_trajectory_data(franka_filepath)
    xarm_positions, xarm_orientations_quat, xarm_times = load_trajectory_data(xarm_filepath)

    # Check if Franka times were loaded
    if franka_times is None:
        print("Error: Franka trajectory file does not contain 'time_from_start' data. Please re-record Franka trajectory.")
        exit()
    
    # Check if xArm times were loaded (should be now)
    if xarm_times is None:
        print("Error: xArm actual trajectory file does not contain 'time_from_start' data. Please re-run xArm playback.")
        exit()

    # Unwrap quaternions for smoother plotting
    franka_orientations_quat_unwrapped = unwrap_quaternions(franka_orientations_quat)
    xarm_orientations_quat_unwrapped = unwrap_quaternions(xarm_orientations_quat)

    # Plot positions
    plot_trajectory_comparison(franka_positions, xarm_positions, 'X-Position', 'X (m)', 0, franka_times, xarm_times)
    plot_trajectory_comparison(franka_positions, xarm_positions, 'Y-Position', 'Y (m)', 1, franka_times, xarm_times)
    plot_trajectory_comparison(franka_positions, xarm_positions, 'Z-Position', 'Z (m)', 2, franka_times, xarm_times)

    # Plot orientations (Quaternion components)
    plot_trajectory_comparison(franka_orientations_quat_unwrapped, xarm_orientations_quat_unwrapped, 'Quaternion X', 'X', 0, franka_times, xarm_times)
    plot_trajectory_comparison(franka_orientations_quat_unwrapped, xarm_orientations_quat_unwrapped, 'Quaternion Y', 'Y', 1, franka_times, xarm_times)
    plot_trajectory_comparison(franka_orientations_quat_unwrapped, xarm_orientations_quat_unwrapped, 'Quaternion Z', 'Z', 2, franka_times, xarm_times)
    plot_trajectory_comparison(franka_orientations_quat_unwrapped, xarm_orientations_quat_unwrapped, 'Quaternion W', 'W', 3, franka_times, xarm_times)

    print("Comparison plots generated and saved as PNG files.")