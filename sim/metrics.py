# This file contains metrics functions used to evaluate a social navigation trial performance.

import numpy as np

def get_path_length(traj):
    # Calculate path length of a trajectory
    return np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1))

def get_path_smoothness(traj):
    # Calculate path smoothness of a trajectory
    # return the average of the absolute angles between the vectors of the trajectory
    norm_product = (np.linalg.norm(traj[1:], axis=1) * np.linalg.norm(traj[:-1], axis=1)) + 1e-9
    angles = np.sum(traj[1:] * traj[:-1], axis=1) / norm_product
    return np.mean(np.abs(np.arccos(angles)))

def get_motion_smoothness(obs_history, dt):
    # Calculate motion smoothness of a trajectory
    # return the avg acceleration of the robot
    time = len(obs_history)
    robot_vel = []
    for t in range(time):
        robot_vel.append(obs_history[t]['robot_vel'])
    robot_vel = np.array(robot_vel)
    robot_acc = np.linalg.norm(robot_vel[1:] - robot_vel[:-1], axis=1) / dt

    return np.mean(robot_acc)

def get_min_ped_dist(obs_history):
    # Calculate minimum distance between robot and pedestrians during the trial
    # return min and avg of the minimum distances

    time = len(obs_history)
    min_dists = []
    for t in range(time):
        robot_pos = obs_history[t]['robot_pos']
        pedestrians_pos = obs_history[t]['pedestrians_pos']
        if pedestrians_pos.shape[0] == 0:
            continue
        min_dist = np.min(np.linalg.norm(pedestrians_pos - robot_pos, axis=1))
        min_dists.append(min_dist)

    return np.min(min_dists), np.mean(min_dists)

def get_min_laser_dist(obs_history):
    # Calculate minimum distance between robot and laser points during the trial
    # return min and avg of the minimum distances

    time = len(obs_history)
    min_dists = []
    for t in range(time):
        robot_pos = obs_history[t]['robot_pos']
        pedestrians_pos = obs_history[t]['laser_pos']
        if len(pedestrians_pos) == 0:
            continue
        else:
            min_dist = np.min(np.linalg.norm(pedestrians_pos - robot_pos, axis=1))
            min_dists.append(min_dist)

    return np.min(min_dists), np.mean(min_dists)