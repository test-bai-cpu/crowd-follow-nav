import os, sys
import csv
import logging
import yaml
import numpy as np
import time
import random
import pandas as pd

from config import get_args, check_args
from sim.simulator import Simulator
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from controller.group_linear_mpc import GroupLinearMPC
from controller.crowd_aware_MPC import CrowdAwareMPC
from controller import mpc_utils
from obs_data_parser import ObsDataParser

#### RL model
import torch
#### -----------------------------------


def set_random_seed(seed):
    seed = seed if seed >= 0 else random.randint(0, 2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


if __name__ == "__main__":
    # configue and logs
    args = get_args()
    set_random_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_fname = os.path.join(args.output_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_fname, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('byteflow').setLevel(logging.WARNING)

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

    ########## Initialize the evaluation results csv file ###########
    # data_file = "ucy_1"
    # data_file = "ucy_2"
    # data_file = "eth_0"
    # data_file = "test"
    # data_file = "all_origin"
    data_file = "synthetic_test2"
    # data_file = "synthetic_train2"
    # data_file = "eth0_left_to_right"
    sim = Simulator(args, f"data/{data_file}.json", logger)
    os.makedirs(os.path.join(sim.output_dir, "evas"), exist_ok=True)
    eva_res_dir = os.path.join(sim.output_dir, "evas", f"{data_file}_{args.exp_name}.csv")
    headers = [
        "case_id", "start_frame", "success", "fail_reason", "navigation_time", "path_length",
        "path_smoothness", "motion_smoothness", "min_ped_dist", "avg_ped_dist",
        "min_laser_dist", "avg_laser_dist"
    ]
    with open(eva_res_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the header row
    #################################################################

    # sim.case_id_list.sort()
    np.random.shuffle(sim.case_id_list)

    mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")
    obs_data_parser = ObsDataParser(mpc_config, args)

    # mpc_horizon = mpc_config.getint('mpc_env', 'mpc_horizon')
    # max_speed = mpc_config.getfloat('mpc_env', 'max_speed')
    max_follow_pos_delta = (mpc_config.getint('mpc_env', 'mpc_horizon') *
                            mpc_config.getfloat('mpc_env', 'max_speed'))
    
    
    # for case_id in [2835]:
    # for case_id in [1068]:
    # for _ in range(5):
    
    ######################### Get the test cases want to check ######################
    # fail_case_file = "exps/failed_cases_noreward.csv"
    # fail_case_df = pd.read_csv(fail_case_file)
    # collision_fail_case_ids = fail_case_df[fail_case_df['fail_reason'] == 'Collision']['case_id'].tolist()
    # time_fail_case_ids = fail_case_df[fail_case_df['fail_reason'] == 'Time']['case_id'].tolist()
    #################################################################################
    
    for case_id in sim.case_id_list:
    # for case_id in [130]:
    # for case_id in collision_fail_case_ids:
        # case_id = random.choice(sim.case_id_list)
        # case_id = 2065
        sim.logger.info(f"Now in the case id: {case_id}")
        obs = sim.reset(case_id)
        done = False
        
        ###### MPC initialization ######
        # mpc = CrowdAwareMPC(mpc_config, args.use_a_omega, args.differential)
        mpc = GroupLinearMPC(mpc_config, args, logger)
        ################################

        time_step = 0
        while not done:
            current_state, target, robot_speed, robot_motion_angle = obs_data_parser.get_robot_state(obs)
            
            ############ Use goal pos as the follow_pos ############
            follow_state = np.array([sim.goal_pos[0], sim.goal_pos[1], 0.0, 0.0])
            follow_state = follow_state.reshape(1, -1)
            ########################################################

            ############ use fixed way to generate a follow state ############
            # follow_state = obs_data_parser.get_follow_state(obs, robot_motion_angle, target) ## follow_state is (4,): pos_x, pos_y, speed, motion_angle
            ########################################################
            
            action_mpc = mpc.get_action(obs, target, follow_state)
            obs, reward, done, info, time_step, info_dict = sim.step(action_mpc, follow_state)

        ############## save the evaluation results to the csv file ##############
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
        #########################################################################