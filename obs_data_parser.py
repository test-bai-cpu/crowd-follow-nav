import numpy as np
from collections import defaultdict

from controller import mpc_utils


class ObsDataParser:
    def __init__(self, config, args):
        self.dt = args.dt
        self.use_a_omega = args.use_a_omega
        
        if config is not None:
            self.configure(config)
        else:
            raise ValueError('Please provide a configuration file')

    def configure(self, config):
        self.mpc_horizon = config.getint('mpc_env', 'mpc_horizon')
        self.max_humans = config.getint('mpc_env', 'max_humans')
        self.group_target_threshold = config.getfloat('mpc_env', 'group_target_threshold')
        self.group_robot_threshold = config.getfloat('mpc_env', 'group_robot_threshold')
        self.group_vel_threshold = config.getfloat('mpc_env', 'group_vel_threshold')

    def get_robot_state(self, obs):
        robot_pos = obs['robot_pos']
        robot_vel = obs['robot_vel']
        robot_th = obs['robot_th'] ## obs['robot_vel'][1] == obs['robot_th']

        robot_speed = robot_vel[0]
        robot_motion_angle = robot_th
        
        if self.use_a_omega:
            current_state = np.array([robot_pos[0], robot_pos[1], robot_speed, robot_motion_angle])
        else:
            current_state = np.array([robot_pos[0], robot_pos[1], robot_motion_angle])

        target = np.array(obs['robot_goal'])
        
        return current_state, target, robot_speed, robot_motion_angle

    def get_human_state(self, obs):
        num_humans = obs["num_pedestrians"]

        if num_humans > self.max_humans:
            # get the closest max_humans state to the robot
            distances_to_humans = np.linalg.norm(obs["pedestrians_pos"] - obs["robot_pos"], axis=1)
            sorted_indices = np.argsort(distances_to_humans)
            nearby_human_pos = obs["pedestrians_pos"][sorted_indices[:self.max_humans]]
            nearby_human_vel = obs["pedestrians_vel"][sorted_indices[:self.max_humans]]
        else: ## padding to max_humans
            nearby_human_pos = np.full((self.max_humans, 2), 1e6)
            nearby_human_vel = np.full((self.max_humans, 2), 1e6)
            if num_humans > 0:
                pedestrians_pos = np.array(obs["pedestrians_pos"]).reshape(num_humans, 2)
                pedestrians_vel = np.array(obs["pedestrians_vel"]).reshape(num_humans, 2)
                nearby_human_pos[:num_humans] = obs["pedestrians_pos"]
                nearby_human_vel[:num_humans] = obs["pedestrians_vel"]
        
        nearby_human_state = np.concatenate((nearby_human_pos, nearby_human_vel), axis=1)
        
        return nearby_human_state
        
        # human_speeds = np.linalg.norm(nearby_human_vel, axis=1)
        # human_motion_angles = np.mod(np.arctan2(nearby_human_vel[:, 1], nearby_human_vel[:, 0]), 2 * np.pi)
        # nearby_human_vel = np.column_stack((human_speeds, human_motion_angles))


    def get_follow_state(self, obs, robot_motion_angle, target):
        ### compute centroid loc and avg speed and motion angle for each group in the observation in obs, group lables are in obs["group_labels"]
        group_data = defaultdict(list)

        for i, label in enumerate(obs["group_labels"]):
            group_data[label].append((obs["pedestrians_pos"][i], obs["pedestrians_vel"][i]))

        group_centroids = {}
        group_vels = {}

        for label, members in group_data.items():
            positions = np.array([m[0] for m in members])
            velocities = np.array([m[1] for m in members])

            speed = np.linalg.norm(velocities, axis=1)
            motion_angle = np.mod(np.arctan2(velocities[:, 1], velocities[:, 0]), 2 * np.pi)
            avg_speed = np.mean(speed)
            avg_motion_angle = mpc_utils.circmean(motion_angle, np.ones(len(motion_angle)))

            centroid = np.mean(positions, axis=0)
            avg_velocity = np.array([avg_speed, avg_motion_angle]) # (speed, motion_angle in radians)
            # avg_velocity = np.mean(velocities, axis=0) ### if using vx and vy as velocity

            group_centroids[label] = centroid
            group_vels[label] = avg_velocity # (speed, motion_angle)

        valid_groups = []
        for group_id, group_vel in group_vels.items():
            # exclude groups that have less than 2 members
            if len(group_data[group_id]) < 2:
                continue
            # exclude groups that are not moving in the same direction as the robot
            angle_diff = mpc_utils.circdiff(group_vel[1], robot_motion_angle)
            if angle_diff >= np.pi / 2:
                continue
            # exclude groups that are too far from the target
            if np.linalg.norm(obs["robot_pos"] - group_centroids[group_id]) > self.group_robot_threshold:
                continue
            # exclude groups that left behind and further way to the goal
            if np.linalg.norm(group_centroids[group_id] - target) - np.linalg.norm(obs["robot_pos"] - target) > self.group_target_threshold:
                continue
            # exclude static groups
            if group_vel[0] < self.group_vel_threshold:
                continue
            
            valid_groups.append(group_id)

        if len(valid_groups) == 0:
            return np.full((1,4), 1e6)
        else:
            nearest_group = min(valid_groups, key=lambda x: np.linalg.norm(group_centroids[x] - obs["robot_pos"]))
            follow_pos = group_centroids[nearest_group]
            follow_vel = group_vels[nearest_group]
            print("First follow pos and vel are: ", follow_pos, follow_vel)

        follow_state = np.concatenate((follow_pos, follow_vel), axis=0).reshape(1, 4)
        
        return follow_state


    def get_follow_state_full_time_version(self, obs, robot_motion_angle):
        ### compute centroid loc and avg speed and motion angle for each group in the observation in obs, group lables are in obs["group_labels"]
        group_data = defaultdict(list)

        for i, label in enumerate(obs["group_labels"]):
            group_data[label].append((obs["pedestrians_pos"][i], obs["pedestrians_vel"][i]))

        group_centroids = {}
        group_vels = {}

        for label, members in group_data.items():
            positions = np.array([m[0] for m in members])
            velocities = np.array([m[1] for m in members])

            speed = np.linalg.norm(velocities, axis=1)
            motion_angle = np.mod(np.arctan2(velocities[:, 1], velocities[:, 0]), 2 * np.pi)
            avg_speed = np.mean(speed)
            avg_motion_angle = mpc_utils.circmean(motion_angle, np.ones(len(motion_angle)))

            centroid = np.mean(positions, axis=0)
            avg_velocity = np.array([avg_speed, avg_motion_angle]) # (speed, motion_angle in radians)
            # avg_velocity = np.mean(velocities, axis=0) ### if using vx and vy as velocity

            group_centroids[label] = centroid
            group_vels[label] = avg_velocity # (speed, motion_angle)

        similar_direction_groups = []
        for group_id, group_vel in group_vels.items():
            # exclude groups that have less than 2 members
            if len(group_data[group_id]) < 2:
                continue
            angle_diff = mpc_utils.circdiff(group_vel[1], robot_motion_angle)
            if angle_diff < np.pi / 2:
                similar_direction_groups.append(group_id)

        if len(similar_direction_groups) == 0:
            follow_pos_in_horizon = None
            follow_vel_in_horizon = None
        else:
            nearest_group = min(similar_direction_groups, key=lambda x: np.linalg.norm(group_centroids[x] - obs["robot_pos"]))
            follow_pos = group_centroids[nearest_group]
            follow_vel = group_vels[nearest_group]
            print("First follow pos and vel are: ", follow_pos, follow_vel)

            #### make follow pos and vel in prediction horizon
            speed = follow_vel[0]
            motion_angle = follow_vel[1]
            follow_pos_in_horizon = np.full((self.mpc_horizon, 2), 1e6)
            follow_vel_in_horizon = np.full((self.mpc_horizon, 2), 1e6)
            follow_pos_in_horizon[0, :] = follow_pos
            follow_vel_in_horizon[0, :] = follow_vel
            for t in range(1, self.mpc_horizon):
                follow_pos[0] += speed * np.cos(motion_angle) * self.dt  # Update x
                follow_pos[1] += speed * np.sin(motion_angle) * self.dt  # Update y
                follow_pos_in_horizon[t, :] = follow_pos  # Store future position
                follow_vel_in_horizon[t, :] = follow_vel  # Store future velocity

        follow_state = np.concatenate((follow_pos_in_horizon, follow_vel_in_horizon), axis=1)
        
        return follow_state
