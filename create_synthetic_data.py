import pandas as pd
import numpy as np
import os


def spawn_group(center_x, center_y, vx_mean, vy_mean, start_frame, direction, goal_right, goal_left, start_area_size, noise_v, duration_frames, fps, person_id_counter):
    # group_size = np.random.randint(5, 11)  # 5 to 10 people
    group_size = np.random.randint(2, 6)  # 2 to 5 people
    
    trajectories = []
    
    for _ in range(group_size):
        # Random initial position within 5x5 square
        x = center_x + np.random.uniform(-start_area_size/2, start_area_size/2)
        y = center_y + np.random.uniform(-start_area_size/2, start_area_size/2)
        
        # Small random noise to velocity
        vx = vx_mean + np.random.uniform(-noise_v, noise_v)
        # vy = vy_mean + np.random.uniform(-noise_v, noise_v)
        vy = vy_mean

        # Generate trajectory
        person_id = person_id_counter
        f = start_frame
        x_new, y_new = x, y
        while 0 <= f < duration_frames:
            # Check if goal is reached
            if (direction == 'right' and x_new >= goal_right) or (direction == 'left' and x_new <= goal_left):
                break
            trajectories.append([f, person_id, x_new, y_new, vx, vy])
            x_new += vx * (1.0 / fps)
            y_new += vy * (1.0 / fps)
            f += 1
        person_id_counter += 1
        
    return person_id_counter, trajectories

def spawn_group_vertical(center_x, center_y, vx_mean, vy_mean, start_frame, direction, goal_right, goal_left, start_area_size, noise_v, duration_frames, fps, person_id_counter):
    ### goal right is goal up
    ### goal left is goal down
    
    group_size = np.random.randint(2, 6)  # 2 to 5 people
    # group_size = np.random.randint(5, 11)  # 5 to 10 people
    
    trajectories = []
    
    for _ in range(group_size):
        # Random initial position within 5x5 square
        x = center_x + np.random.uniform(-start_area_size/2, start_area_size/2)
        y = center_y + np.random.uniform(-start_area_size/2, start_area_size/2)
        
        # Small random noise to velocity
        vx = vx_mean
        # vy = vy_mean + np.random.uniform(-noise_v, noise_v)
        vy = vy_mean + np.random.uniform(-noise_v, noise_v)

        # Generate trajectory
        person_id = person_id_counter
        f = start_frame
        x_new, y_new = x, y
        while 0 <= f < duration_frames:
            # Check if goal is reached
            if (direction == 'up' and y_new >= goal_right) or (direction == 'down' and y_new <= goal_left):
                break
            trajectories.append([f, person_id, x_new, y_new, vx, vy])
            x_new += vx * (1.0 / fps)
            y_new += vy * (1.0 / fps)
            f += 1
        person_id_counter += 1
        
    return person_id_counter, trajectories

def generate_synthetic_data():
    # Simulate data, forming into a synthetic dataset

    fps = 10
    frames_per_group_base = 50
    n_groups = 100  # pairs of groups
    goal_right = 20
    goal_left = 0
    start_area_size = 5  # 5x5 square
    noise_v = 0.05  # small velocity noise
    
    duration_frames = frames_per_group_base * n_groups
    trajectories = []
    person_id_counter = 0
    
    np.random.seed(42)  # for reproducibility

    for group_idx in range(n_groups):
        frames_per_group = np.random.randint(frames_per_group_base - 10, frames_per_group_base + 10) # every 4-6 seconds a new group
        start_frame = group_idx * frames_per_group
        
        ################### The horizontal flow, to generate traj_1.csv ###################
        # # Group 1: From (0,5), rightward
        # person_id_counter, group_trajectories = spawn_group(0, 5, 0.9, 0, start_frame, 'right', goal_right, goal_left, start_area_size, noise_v, duration_frames, fps, person_id_counter)
        # trajectories.extend(group_trajectories)
        # # Group 2: From (20,10), leftward
        # person_id_counter, group_trajectories = spawn_group(20, 10, -0.9, 0, start_frame, 'left', goal_right, goal_left, start_area_size, noise_v, duration_frames, fps, person_id_counter)
        # trajectories.extend(group_trajectories)
        ############################################################
        
        ################### The vertical flow, to generate traj_2.csv ###################
        # Group 1: From (0,5), rightward
        person_id_counter, group_trajectories = spawn_group_vertical(5, 0, 0, 0.9, start_frame, 'up', goal_right, goal_left, start_area_size, noise_v, duration_frames, fps, person_id_counter)
        trajectories.extend(group_trajectories)
        # Group 2: From (20,10), leftward
        person_id_counter, group_trajectories = spawn_group_vertical(10, 20, 0, -0.9, start_frame, 'down', goal_right, goal_left, start_area_size, noise_v, duration_frames, fps, person_id_counter)
        trajectories.extend(group_trajectories)
        ############################################################
        
        
    columns = ["frame_id", "person_id", "x", "y", "vx", "vy"]
    data = pd.DataFrame(trajectories, columns=columns)
    data = data.sort_values(by=["frame_id", "person_id"]).reset_index(drop=True)
    data["frame_id"] = data["frame_id"].round().astype(int)
    data_folder = "sim/synthetic_data"
    os.makedirs(data_folder, exist_ok=True)
    data.to_csv(f"{data_folder}/traj_6.csv", index=False)
    
generate_synthetic_data()