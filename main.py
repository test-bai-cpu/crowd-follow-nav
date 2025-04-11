import os, sys
import csv
import logging
import yaml
import numpy as np
import random
import time

from config import get_args, check_args
from sim.simulator import Simulator
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from controller.group_linear_mpc import GroupLinearMPC
from controller.crowd_aware_MPC import CrowdAwareMPC
from controller import mpc_utils
from obs_data_parser import ObsDataParser

#### RL model
import torch
from torch.utils.tensorboard import SummaryWriter
from rl.rl_agent import SAC
from rl.trainer import ContinuousSACTrainer
from rl.utils import load_config
#### -----------------------------------


#### RL model
def preprocess_rl_obs(obs, current_state, robot_vx, robot_vy, goal_pos):
    """ img_obs: A Numpy array with (max_human, 4) in float32.
        Process it into torch tensor with (bs, max_humna*4) in float32.
    """
    obs = obs.copy()
    current_state = current_state.copy()
    current_pos = current_state[:2].reshape(1, -1)
    obs[:, :2] = obs[:, :2] - current_pos
    obs[obs > 1e4] = 0

    obs[:, 2] = obs[:, 2] - robot_vx
    obs[:, 3] = obs[:, 3] - robot_vy

    goal_pos = np.array(goal_pos).reshape(1, -1)
    goal_pos = goal_pos - current_pos
    goal_vx_vy = np.array([-robot_vx, -robot_vy]).reshape(1, -1)
    obs = obs.reshape(1, -1)
    obs = np.concatenate([goal_pos, goal_vx_vy, obs], axis=1)
    return obs

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
    # data_file = "all"
    data_file = "eth0_left_to_right"
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

    ######################### RL model #####################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_config = load_config("rl_config.yaml")
    result_dir = os.path.join(rl_config["result_dir"], args.exp_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    rl_agent = SAC(rl_config["state_shape"], rl_config["action_shape"],
                   rl_config["latent_dim"], device)
    rl_trainer = ContinuousSACTrainer(rl_agent, result_dir, rl_config)
    train_info = {}
    # max_follow_pos_delta = rl_config["max_follow_pos_delta"]

    tb_writer = SummaryWriter(f"{rl_config['result_dir']}/logs/{args.exp_name}/{int(time.time())}")
    logger.info(f"RL result directory: {result_dir}")
    ########################################################################
    # sim.case_id_list.sort()
    np.random.shuffle(sim.case_id_list)

    mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")
    obs_data_parser = ObsDataParser(mpc_config, args)

    # mpc_horizon = mpc_config.getint('mpc_env', 'mpc_horizon')
    # max_speed = mpc_config.getfloat('mpc_env', 'max_speed')
    max_follow_pos_delta = (mpc_config.getint('mpc_env', 'mpc_horizon') *
                            mpc_config.getfloat('mpc_env', 'max_speed'))

    for case_id in sim.case_id_list:
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
            robot_vx = robot_speed * np.cos(robot_motion_angle)
            robot_vy = robot_speed * np.sin(robot_motion_angle)
            nearby_human_state = obs_data_parser.get_human_state(obs) ## padding to max_humans, padding with 1e6 (for pos and vel). Human_state is (n, 4): pos_x, pos_y, vel_x, vel_y

            ############ RL model output the follow_pos ############
            rl_trainer.global_step += 1
            rl_obs = preprocess_rl_obs(nearby_human_state, current_state, robot_vx, robot_vy, sim.goal_pos) ## TODO: can move it outside the loop?
            if rl_trainer.is_learning_starts():
                rl_actions, _, entropies = rl_agent.get_action(torch.FloatTensor(rl_obs).to(device))
                rl_actions = rl_actions.cpu().detach().numpy()
            else:
                # rl_obs = preprocess_rl_obs(nearby_human_state, current_state, robot_vx, robot_vy, sim.goal_pos)
                rl_actions = rl_agent.random_actions()

            # Rescale actions. rl_actions is (1, 4): pos_x, pos_y, vel_x, vel_y, and they are all relative values to the robot, both pos and vel
            follow_pos = rl_actions[0, :2].copy()
            # follow_vel = rl_actions[0, 2:].copy()

            ## Now rerange the follow_pos and follow_vel
            # follow_pos = (follow_pos + 1) * (max_follow_pos_delta + max_follow_pos_delta) / 2 - max_follow_pos_delta     # Since max_follow_pos_delta > 0
            follow_pos = follow_pos * max_follow_pos_delta     # Since max_follow_pos_delta > 0
            # revert the relative pos to global pos
            follow_pos = follow_pos + current_state[:2]

            # follow_vel = (follow_vel + 1) * (mpc.max_speed + mpc.max_rev_speed) / 2 - mpc.max_rev_speed     # Since max_rev_speed > 0
            # follow_vel = follow_vel + np.array([robot_vx, robot_vy])

            # follow_speed = np.linalg.norm(follow_vel)
            # follow_motion_angle = np.mod(np.arctan2(follow_vel[1], follow_vel[0]), 2 * np.pi)

            # follow_state = np.array([follow_pos[0], follow_pos[1], follow_speed, follow_motion_angle])
            follow_state = np.array([follow_pos[0], follow_pos[1], 0, 0])
            follow_state = follow_state.reshape(1, -1)
            #########################################################

            # follow_state = obs_data_parser.get_follow_state(obs, robot_motion_angle, target) ## follow_state is (4,): pos_x, pos_y, speed, motion_angle


            for mpc_steps_in_one_follow_state in range(10):
                ###### MPC generate action ######
                # action_mpc, _ = mpc.get_action(obs, current_state, target, nearby_human_state, follow_state)
                action_mpc = mpc.get_action(obs, target, follow_state)
                ################################

                obs, reward, done, info, time_step, info_dict = sim.step(action_mpc, follow_state)
                if done == True:
                    break

            #################### RL model output the follow_pos #############################
            current_state, target, robot_speed, robot_motion_angle = obs_data_parser.get_robot_state(obs)
            robot_vx = robot_speed * np.cos(robot_motion_angle)
            robot_vy = robot_speed * np.sin(robot_motion_angle)
            next_nearby_human_state = obs_data_parser.get_human_state(obs) ## padding to max_humans, padding with 0 (for pos and vel). Human_state is (n, 4): pos_x, pos_y, vel_x, vel_y

            ## TODO: check do I need to pass device here?
            next_rl_obs = preprocess_rl_obs(next_nearby_human_state, current_state, robot_vx, robot_vy, sim.goal_pos)

            rl_reward = np.array([reward])
            rl_done = np.array([done])
            rl_trainer.add_to_buffer(rl_obs, next_rl_obs, rl_actions, rl_reward, rl_done, [{}])
            rl_trainer.update_episode_info(rl_reward)
            if "reach_goal_reward" in info_dict:
                rl_trainer.reporter["reach_goal_reward"].append(info_dict["reach_goal_reward"])
                rl_trainer.reporter["reach_goal_reward_dense"].append(info_dict["reach_goal_reward_dense"])
                rl_trainer.reporter["group_matching_reward"].append(info_dict["group_matching_reward"])

            if rl_done.any():
                rl_infos = {"is_success": np.array([info_dict["reach_goal_reward"] > 0])}   # info is a boolean value representing whether the robot reaches the target
                rl_trainer.record_episode_info(rl_done, rl_infos)

            if rl_trainer.is_train_model():
                train_info = rl_trainer.train_model()

            rl_trainer.report_progress(tb_writer, train_info)
            rl_trainer.save_model()
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
