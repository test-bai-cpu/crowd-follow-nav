import os, sys
import csv
import logging
import yaml
import numpy as np
from collections import defaultdict

from config import get_args, check_args
from sim.simulator import Simulator
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from controller.crowd_aware_MPC import CrowdAwareMPC
from controller import mpc_utils


def parse_obs_data(args, obs, config):
    mpc_horizon = config.getint('mpc_env', 'mpc_horizon')
    dt = args.dt
    
    robot_pos = obs['robot_pos']
    robot_vel = obs['robot_vel']
    robot_th = obs['robot_th']

    robot_speed = robot_vel[0]
    robot_motion_angle = robot_th
    
    if args.use_a_omega:
        current_state = np.array([robot_pos[0], robot_pos[1], robot_speed, robot_motion_angle])
    else:
        current_state = np.array([robot_pos[0], robot_pos[1], robot_motion_angle])

    target = np.array(obs['robot_goal'])
    
    # max_humans = config.getint('mpc_env', 'max_humans')
    num_humans = obs["num_pedestrians"]
    
    if num_humans == 0:
        nearby_human_pos = None
        nearby_human_vel = None
        return current_state, target, None, None, None, None, robot_speed, robot_motion_angle
    # elif num_humans > max_humans:
    #     # get the closest max_humans to the robot
    #     distances_to_humans = np.linalg.norm(obs["pedestrians_pos"] - obs["robot_pos"], axis=1)
    #     sorted_indices = np.argsort(distances_to_humans)
    #     nearby_human_pos = obs["pedestrians_pos"][sorted_indices[:max_humans]]
    #     nearby_human_vel = obs["pedestrians_vel"][sorted_indices[:max_humans]]

    # change human vel from vx, vy to speed, motion_angle
    nearby_human_pos = obs["pedestrians_pos"]
    human_speeds = np.linalg.norm(obs["pedestrians_vel"], axis=1)
    human_motion_angles = np.mod(np.arctan2(obs["pedestrians_vel"][:, 1], obs["pedestrians_vel"][:, 0]), 2 * np.pi)
    nearby_human_vel = np.column_stack((human_speeds, human_motion_angles))
    
    ## When use the original vx and vy
    # nearby_human_pos = obs["pedestrians_pos"]
    # nearby_human_vel = obs["pedestrians_vel"]

    ### compute centroid loc and avg speed and motion angle for each group in the observation
    ### in obs, group lables are in obs["group_labels"]
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
        # avg_velocity = np.mean(velocities, axis=0)
        
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
        follow_pos_in_horizon = np.zeros((mpc_horizon, 2))
        follow_vel_in_horizon = np.zeros((mpc_horizon, 2))
        follow_pos_in_horizon[0, :] = follow_pos
        follow_vel_in_horizon[0, :] = follow_vel
        for t in range(1, mpc_horizon):
            follow_pos[0] += speed * np.cos(motion_angle) * dt  # Update x
            follow_pos[1] += speed * np.sin(motion_angle) * dt  # Update y
            follow_pos_in_horizon[t, :] = follow_pos  # Store future position
            follow_vel_in_horizon[t, :] = follow_vel  # Store future velocity
        
        # print("####################")
        # print(follow_pos, follow_vel)
        # print(follow_pos_in_horizon, follow_vel_in_horizon)
        # print("####################")

    return current_state, target, nearby_human_pos, nearby_human_vel, follow_pos_in_horizon, follow_vel_in_horizon, robot_speed, robot_motion_angle


if __name__ == "__main__":
    # configue and logs
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_fname = os.path.join(args.output_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_fname, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    check_args(args, logger)

    # which datasets to preload
    yaml_stream = open(args.dset_file, "r")
    yaml_dict = yaml.safe_load(yaml_stream)
    dsets = yaml_dict["datasets"]
    flags = yaml_dict["flags"]
    if not len(dsets) == len(flags):
        logger.error("datasets file - number of datasets and flags are not equal!")
        raise Exception("datasets file - number of datasets and flags are not equal!")
    
    envs_arg = []
    for i in range(len(dsets)):
        dset = dsets[i]
        flag = flags[i]
        envs_arg.append((dset, flag))
    args.envs = envs_arg

    
    # sim = Simulator(args, 'data/eth_0.json', logger)
    data_file = "ucy_0"
    sim = Simulator(args, f"data/{data_file}.json", logger)
    os.makedirs(os.path.join(sim.output_dir, "evas"), exist_ok=True)
    eva_res_dir = os.path.join(sim.output_dir, "evas", f"{data_file}.csv")
    headers = [
        "case_id", "start_frame", "success", "fail_reason", "navigation_time", "path_length",
        "path_smoothness", "motion_smoothness", "min_ped_dist", "avg_ped_dist",
        "min_laser_dist", "avg_laser_dist"
    ]
    with open(eva_res_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the header row
    
    for case_id in sim.case_id_list:
        # if case_id != 19:
        #     continue
        sim.logger.info(f"Now in the case id: {case_id}")
        obs = sim.reset(case_id)
        done = False
        mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")
        mpc = CrowdAwareMPC(mpc_config, args.use_a_omega)
        time_step = 0
        while not done:
            current_state, target, nearby_human_pos, nearby_human_vel, follow_pos, follow_vel, robot_speed, robot_motion_angle = parse_obs_data(args, obs, mpc_config)
            #### RL model output the follow_pos
            action_mpc, _ = mpc.get_action(current_state, target, nearby_human_pos, nearby_human_vel, follow_pos, follow_vel)
            print("--- action ---")
            print("use a_omega is: ", args.use_a_omega, ", and action_mpc: ", action_mpc)
            ## action a, omega to vx, vy
            ## obs_robot_vel is vx, vy, now I have action a, omega, I want to compute the new robot_vel in vx, vy
            # robot_speed_new = robot_speed + action_mpc[0] * args.dt
            # robot_motion_angle_new = robot_motion_angle + action_mpc[1] * args.dt
            # action = np.array([robot_speed_new * np.cos(robot_motion_angle_new), robot_speed_new * np.sin(robot_motion_angle_new)])
            # print("vxvy: ", action)
            # print("speed: ", robot_speed_new, "motion_angle: ", robot_motion_angle_new)
            # if time_step == 1593:
            #     print("Now checking the obs value")
            obs, reward, done, info, time_step = sim.step(action_mpc, follow_pos, follow_vel)
            
            print("time step: ", time_step)

        result_dict = sim.evaluate(output=True)
        with open(eva_res_dir, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                case_id,
                sim.start_frame,
                result_dict["success"],
                result_dict["fail_reason"],
                result_dict["navigation_time"],
                result_dict["path_length"],
                result_dict["path_smoothness"],
                result_dict["motion_smoothness"],
                result_dict["min_ped_dist"],
                result_dict["avg_ped_dist"],
                ])  # Write the header row

    # sim = Simulator(args, 'data/eth_0.json', logger)
    # agent = PedNoPredMPC(args, logger)
    # obs = sim.reset(100)
    # done = False
    # while not done:
    #     action = agent.act(obs)
    #     obs, reward, done, info = sim.step(action)
    # sim.evaluate(output=True)
