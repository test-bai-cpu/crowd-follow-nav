import os, sys
import csv
import logging
import yaml
import numpy as np

from config import get_args, check_args
from sim.simulator import Simulator
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from controller.crowd_aware_MPC import CrowdAwareMPC
from controller import mpc_utils
from obs_data_parser import ObsDataParser


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

    # data_file = "ucy_1"
    data_file = "1024-1"
    sim = Simulator(args, f"data/{data_file}.json", logger)
    
    print(f"In dataset{data_file}, there are {len(sim.case_id_list)} cases in total.")
    
    print(sim.case_id_list)
    
    ### order the case_id_list
    sim.case_id_list.sort()
    for case_id in sim.case_id_list:
        if case_id in [0,1,2,3]:
            continue
        sim.logger.info(f"Now in the case id: {case_id}")
        obs = sim.reset(case_id)
        done = False
        action = np.array([0, 0])
        
        while not done:
            obs, reward, done, info, time_step = sim.step(action)
            print("time step: ", time_step)