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

#### RL model
import time
import torch
from torch.utils.tensorboard import SummaryWriter
# from rl.rl_agent import SAC
# from rl.trainer import ContinuousSACTrainer
# from rl.utils import load_config
#### -----------------------------------


#### RL model
def preprocess_rl_obs(obs, device):
    """ img_obs: A Numpy array with (max_human, 4) in float32.
        Process it into torch tensor with (bs, max_humna*4) in float32.
    """
    return torch.FloatTensor(obs.reshape(1, -1)).to(device).type(torch.float)
#### -----------------------------------


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

    ########## Initialize the evaluation results csv file ###########
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
    #################################################################
    
    ######################### RL model #####################################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # rl_config = load_config("configs/rl_config.yaml")
    # result_dir = rl_config["result_dir"]
    # rl_agent = SAC(rl_config["state_shape"], rl_config["action_shape"],
    #                rl_config["latent_dim"], device)
    # rl_trainer = ContinuousSACTrainer(rl_agent, result_dir, rl_config)
    # train_info = {}
    # max_follow_pos_delta = rl_config["max_follow_pos_delta"]

    # writer = SummaryWriter(f"{result_dir}/logs/{int(time.time())}")
    ########################################################################

    for case_id in sim.case_id_list:
        # if case_id != 19:
        #     continue
        sim.logger.info(f"Now in the case id: {case_id}")
        obs = sim.reset(case_id)
        done = False
        mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")
        obs_data_parser = ObsDataParser(mpc_config, args)
        mpc = CrowdAwareMPC(mpc_config, args.use_a_omega, args.differential)
        time_step = 0
        while not done:
            current_state, target, robot_speed, robot_motion_angle = obs_data_parser.get_robot_state(obs)
            nearby_human_state = obs_data_parser.get_human_state(obs) ## padding to max_humans
            # TODO: padding and concate the nearby_human_pos and nearby_human_vel to max_humans -> nearby_human_state
            
            ############ RL model output the follow_pos ############
            # rl_trainer.global_step += 1
            # if rl_trainer.is_learning_starts():
            #     rl_obs = preprocess_rl_obs(nearby_human_state)
            #     rl_actions, _, entropies = rl_agent.get_action(rl_obs)
            #     rl_actions = rl_actions.cpu().detach().numpy()
            # else:
            #     rl_actions = rl_agent.random_actions()

            # # Rescale actions
            # follow_pos = rl_actions[0, :2]
            # follow_vel = rl_actions[0, 2:]

            # follow_pos = (follow_pos + 1) * (max_follow_pos_delta + max_follow_pos_delta) / 2 - max_follow_pos_delta     # Since max_follow_pos_delta > 0
            # follow_vel = (follow_vel + 1) * (mpc.max_speed + mpc.max_rev_speed) / 2 - mpc.max_rev_speed     # Since max_rev_speed > 0
            #########################################################

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

            #################### RL model output the follow_pos #############################
            # (current_state, target, nearby_human_pos, nearby_human_vel,
            #  follow_pos, follow_vel, robot_speed, robot_motion_angle) = \
            #      parse_obs_data(args, obs, mpc_config)
            # next_nearby_human_state = None
            
            # # next_rl_obs = preprocess_rl_obs(next_nearby_human_state)
            # next_rl_obs = next_nearby_human_state.reshape(1, -1)
            # rl_reward = np.array([[reward]])
            # rl_done = np.array([[done]])
            # rl_trainer.add_to_buffer(rl_obs, next_rl_obs, rl_actions, rl_reward, rl_done, [{}])
            # rl_trainer.update_episode_info(rl_reward)

            # if done.any():
            #     infos = {}
            #     rl_trainer.record_episode_info(rl_done, infos)

            # if rl_trainer.is_train_model():
            #     train_info = rl_trainer.train_model()

            # rl_trainer.report_progress(writer, train_info)
            # rl_trainer.save_model()
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

    # sim = Simulator(args, 'data/eth_0.json', logger)
    # agent = PedNoPredMPC(args, logger)
    # obs = sim.reset(100)
    # done = False
    # while not done:
    #     action = agent.act(obs)
    #     obs, reward, done, info = sim.step(action)
    # sim.evaluate(output=True)
