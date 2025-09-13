import os, sys

sys.path.append(os.path.abspath("./crowdattn"))

import csv
import logging
import yaml
import numpy as np
import time
import random
import pandas as pd
import pickle

from config import get_args, check_args
from sim.simulator import Simulator
from controller.group_linear_mpc import GroupLinearMPC
from controller import mpc_utils
from obs_data_parser import ObsDataParser

from sim.crowd_attn_rl import CrowdAttnRL
import torch


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
    
    args.group = True
    args.record = True
    args.react = True
    args.history = True
    args.animate = True
    args.goal_radius = 2.0

    dataset_name = "eth"       # "syn" / "eth"
    
    args.exp_name = f"e001_attl_{dataset_name}_orca"
    # args.exp_name = f"e001_attl_{dataset_name}_orcanorobot"
    # args.exp_name = f"e001_attl_{dataset_name}_sfm"
    # args.exp_name = f"e001_attl_{dataset_name}_sfmnorobot"
    
    # args.exp_name = f"e001_attl_{dataset_name}_noreact"       # args.react = False
    # args.react = False
    
    
    if dataset_name == "syn":
        args.dset_file = "datasets_syn.yaml"        # eth: datasets.yaml, syn: datasets_syn.yaml
    else:
        args.dset_file = "datasets.yaml"
    args.collision_radius = 0.5
    # args.output_dir = f"exps/results_attl_41665_speed1/{args.exp_name}"
    args.output_dir = f"exps/results_quality/attn/{args.exp_name}"
    
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
    if args.dset_file == "datasets.yaml":
        data_file = "eth_ucy_test"
    elif args.dset_file == "datasets_syn.yaml": # synthetic datasets
        data_file = "synthetic_test"

    print("<<<< The args.react are: ", args.react, args)
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

    sim.case_id_list.sort()

    mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")
    obs_data_parser = ObsDataParser(mpc_config, args)
    
    # for case_id in [573]:
    for case_id in [1988, 1936, 1904, 1747, 1663]:
    # for case_id_index in range(500):
        # case_id = random.choice(sim.case_id_list)
        sim.logger.info(f"Now in the case id: {case_id}")
        obs = sim.reset(case_id)
        done = False
        
        case_info = sim.get_case_info()
        # set up the prediction model checkpoint path

        # if (case_info['env_name'] == 'eth') and (case_info['env_flag'] == 0):
        #     sgan_model_path = "sgan/models/sgan-models/eth_" + str(args.future_steps) + "_model.pt"
        # elif (case_info['env_name'] == 'eth') and (case_info['env_flag'] == 1):
        #     sgan_model_path = "sgan/models/sgan-models/hotel_" + str(args.future_steps) + "_model.pt"
        # elif (case_info['env_name'] == 'ucy') and (case_info['env_flag'] == 0):
        #     sgan_model_path = "sgan/models/sgan-models/zara1_" + str(args.future_steps) + "_model.pt"
        # elif (case_info['env_name'] == 'ucy') and (case_info['env_flag'] == 1):
        #     sgan_model_path = "sgan/models/sgan-models/zara2_" + str(args.future_steps) + "_model.pt"
        # elif (case_info['env_name'] == 'ucy') and (case_info['env_flag'] == 2):
        #     sgan_model_path = "sgan/models/sgan-models/univ_" + str(args.future_steps) + "_model.pt"
        # else:
        #     sgan_model_path = "sgan/models/sgan-models/eth_" + str(args.future_steps) + "_model.pt"
        
        
        args.future_steps = 5
        # agent = CrowdAttnRL(args, logger, 'crowdattn/trained_models', ckpt='23576693.pt')
        agent = CrowdAttnRL(args, logger, 'crowdattn/trained_models', ckpt='41665.pt')
        
        time_step = 0
        while not done:
            
            action = agent.act(obs, done)
            vx = action[0]
            vy = action[1]
            action_speed = float(np.hypot(vx, vy))
            
            theta_des = np.arctan2(vy, vx)
            theta_cur = sim.robot_th  # current heading
            angle_err = mpc_utils.circdiff(theta_des, theta_cur)

            k_w = 1.0
            w_max = mpc_config.getfloat('mpc_env', 'max_rot_degrees') * sim.dt
            action_theta = float(np.clip(k_w * angle_err, -w_max, w_max))
            action_vw = np.array([action_speed, action_theta], dtype=np.float32)
            
            obs, reward, done, info, time_step, info_dict = sim.step(action_vw)
            
            if args.animate and not args.paint_boundary:
                frame = sim.get_latest_render_frame()
                sim.update_latest_render_frame(frame)
            

        ################# save the robot path and human path #############################
        save_filename = f"{data_file}_{args.exp_name}.pkl"
        save_filepath = os.path.join(sim.output_dir, "evas", save_filename)
        
        existing_data = {}
        if os.path.exists(save_filepath):
            try:
                with open(save_filepath, "rb") as f:
                    existing_data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                existing_data = {}
        
        existing_data[case_id] = sim.save_all_traj.copy()
        
        with open(save_filepath, "wb") as f:
            pickle.dump(existing_data, f)
            logger.info(f"Case {case_id} trajectory appended to {save_filepath}")
        #################################################################################

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