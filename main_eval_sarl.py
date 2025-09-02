import os, sys

# sys.path.append(os.path.abspath("/home/yufei/research/crowd-follow-nav"))
sys.path.append(os.path.abspath("./baseline_code/CrowdNav"))

# from baseline_code.CrowdNav.crowd_sim.envs.utils.state import ObservableState
# from baseline_code.CrowdNav.crowd_sim.envs.utils import state
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils import state

print(">>> sys.path =", sys.path)

import csv
import copy
import logging
import yaml
import numpy as np
import pickle
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

#### SARL
import sarl.sarl_sim_utils as sarl_utils


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


def parse_robot_state_for_sarl(obs, sim, env_config, mpc_config):
    px = obs["robot_pos"][0]
    py = obs["robot_pos"][1]
    vx = obs["robot_vel"][0]
    vy = obs["robot_vel"][1]
    radius = env_config.getfloat('robot', 'radius')
    gx = sim.goal_pos[0]
    gy = sim.goal_pos[1]
    v_pref = mpc_config.getfloat('mpc_env', 'pref_speed')
    theta = obs["robot_th"]
    return px, py, vx, vy, radius, gx, gy, v_pref, theta


class Explorer(object):
    def __init__(self, sim, policy, device, memory=None, gamma=None, target_policy=None,
                 env_config={}, mpc_config={}):
        # self.env = sim
        # self.robot = policy
        self.sim = sim
        self.policy = policy
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        self.env_config = env_config
        self.mpc_config = mpc_config

        self.n_transitions = 0
        self.hist_complete = []
        self.episode_total_rewards = []
        self.max_humans = mpc_config.getint('mpc_env', 'max_humans')
        self.max_human_distance = mpc_config.getfloat('mpc_env', 'max_human_distance')

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def _preprocess_human_states(self, robot_fullstate):
        num_humans = len(robot_fullstate.human_states)

        if num_humans == 0:
            robot_fullstate.human_states = [ObservableState(100, 100, 1, 1, 0)
                                            for _ in range(self.max_humans)]

        else:
            robot_pos = np.array(robot_fullstate.self_state.position)     # (px, py)

            human_index, human_pos = [], []
            for i, human_state in enumerate(robot_fullstate.human_states):
                human_pos.append(human_state.position)                # (px, py)
                human_index.append(i)
            human_pos = np.stack(human_pos, axis=0)
            human_index = np.array(human_index)

            # Filter by distance threshold
            distances_to_humans = np.linalg.norm(human_pos - robot_pos, axis=1)
            within_threshold = distances_to_humans < self.max_human_distance
            filtered_pos = human_pos[within_threshold]
            filtered_index = human_index[within_threshold]

            # Limit/padding to max_humans
            num_filtered = filtered_pos.shape[0]
            if num_filtered > self.max_humans:
                # get the closest max_humans state to the robot
                sorted_indices = np.argsort(np.linalg.norm(filtered_pos - robot_pos, axis=1))
                filtered_index = filtered_index[sorted_indices[:self.max_humans]]
            nearby_human_states = [robot_fullstate.human_states[index]
                                   for index in filtered_index]
            robot_fullstate.human_states = nearby_human_states

            # Padding to max_humans
            if len(robot_fullstate.human_states) < self.max_humans:
                for _ in range(self.max_humans - num_filtered):
                    robot_fullstate.human_states.append(ObservableState(0, 0, 0, 0, 0))

        return robot_fullstate

    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False):
        self.policy.set_phase(phase)
        for i in range(k):
            case_id = random.choice(self.sim.case_id_list)
            # case_id = 1791
            sim.logger.info(f"In imitation learning: {imitation_learning}, Running episode {i + 1}/{k} in {phase} phase. In case_id {case_id}")

            obs = self.sim.reset(case_id)
            done = False

            states, actions, rewards = [], [], []

            while not done:
                # (SARL) Collect agent observation
                sarl_obs = sarl_utils.process_obs_to_sarl(obs)
                px, py, vx, vy, radius, gx, gy, v_pref, theta = \
                    parse_robot_state_for_sarl(obs, sim, self.env_config, self.mpc_config)

                robot_fullstate = sarl_utils.get_robot_fullstate(
                    sarl_obs, px, py, vx, vy, radius, gx, gy, v_pref, theta)

                # Limite to maximum K humans nearby the robot
                # (check `ObsDataParser.get_human_state(...)` in obs_data_parser.py)
                robot_fullstate = self._preprocess_human_states(robot_fullstate)

                # if not imitation_learning:
                #     print("Dist =", np.linalg.norm(
                #         robot_fullstate.human_states[0].position -
                #         np.array(robot_fullstate.self_state.position)))

                # (SARL) Estimate the action
                action = self.policy.predict(robot_fullstate)
                # print("<<<<<<<<<<<<: ", action)
                if imitation_learning: ## in ORCA
                    v = np.linalg.norm([action.vx, action.vy])
                    move_dir = np.arctan2(vy, vx)
                    omega = sarl_utils.wrapToPi(move_dir - sim.robot_th) / sim.dt
                    sim_action = [v, omega]
                else: ## in SARL
                    sim_action = np.array([action.v, action.r])
                    # print(">>> in Training, sim_action =", sim_action)
                obs, _, done, info, time_step, info_dict = sim.step(sim_action)

                ############## get reward from obs ################
                pedestrians_pos = obs['pedestrians_pos'].copy()
                robot_pos = obs['robot_pos'].copy()
                if pedestrians_pos.shape[0] == 0:
                    dmin = np.inf  # or some default large value
                else:
                    dmin = np.min(np.linalg.norm(pedestrians_pos - robot_pos, axis=1))
                discomfort_dist = self.env_config.getfloat('reward', 'discomfort_dist')
                if info == True:
                    reward = self.env_config.getfloat('reward', 'success_reward')
                elif sim.fail_reason == "Time":
                    reward = 0
                elif sim.fail_reason == "Collision":
                    reward = self.env_config.getfloat('reward', 'collision_penalty')
                    print("Collision penalty: ", reward)
                elif dmin < discomfort_dist:
                    discomfort_penalty_factor = self.env_config.getfloat('reward', 'discomfort_penalty_factor')
                    reward = (dmin - discomfort_dist) * discomfort_penalty_factor * self.sim.dt
                    print("Discomfort penalty: ", reward)
                else:
                    reward = info_dict['reach_goal_reward_dense']
                    # print("Reach goal dense: ", reward)
                # print("<<<<<. ", reward)
                ####################################################

                self.n_transitions += 1
                states.append(self.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                # Report
                if self.n_transitions % 5000 == 0:
                    if tb_writer is not None:
                        tb_writer.add_scalar(f"rollout/total_reward", np.mean(self.episode_total_rewards), self.n_transitions)
                        tb_writer.add_scalar(f"rollout/hist_complete", np.mean(self.hist_complete), self.n_transitions)
                    self.hist_complete = []
                    self.episode_total_rewards = []

            is_success = np.array([info])
            self.hist_complete += is_success[done].tolist()
            self.episode_total_rewards.append(np.sum(np.array(rewards)[done]))

            sim.logger.info(f"Episode success: {info} | EP reward: {np.sum(np.array(rewards)):.2f} | EP length: {self.n_transitions}")


            if update_memory:
                reach_goal = is_success
                collision = not reach_goal and self.sim.fail_reason == "Collision"
                if reach_goal or collision:
                    self.update_memory(states, actions, rewards, v_pref, imitation_learning)

            # Assume imitation 70_000 and rl 230_000 transitions
            # if self.n_transitions >= 70000 and imitation_learning:
            #     sim.logger.info("Collect more than 70000 transitions.")
            #     break
            # elif self.n_transitions >= 300000:
            #     sim.logger.info("Collect more than 300000 transitions.")
            #     break

        return case_id

    def update_memory(self, states, actions, rewards, v_pref, imitation_learning=False):
        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.sim.dt * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.sim.dt * v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.sim.dt * v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # state with shape (# of humans, len(state))
            self.memory.push((state, value))


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
    # args.animate = True
    args.group = True
    args.record = True
    args.react = True
    
    dataset_name = "eth"       # "syn"
    
    # args.exp_name = f"e104_sarl_{dataset_name}_orca"
    # args.exp_name = f"e104_sarl_{dataset_name}_orcanorobot"
    # args.exp_name = f"e104_sarl_{dataset_name}_sfm"
    # args.exp_name = f"e104_sarl_{dataset_name}_sfmnorobot"
    
    args.exp_name = f"e104_sarl_{dataset_name}_noreact"       # args.react = False
    args.react = False
    
    
    if dataset_name == "syn":
        args.dset_file = "datasets_syn.yaml"        # eth: datasets.yaml, syn: datasets_syn.yaml
    else:
        args.dset_file = "datasets.yaml"
    args.collision_radius = 0.5
    args.output_dir = f"exps/results_sarl_20250902/{args.exp_name}"

    total_eval_episodes = 500

    set_random_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

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

    mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")

    ########## Initialize the evaluation results csv file ###########
    # data_file = "eth_ucy_train"
    # data_file = "synthetic_train"
    if args.dset_file == "datasets.yaml":
        data_file = "eth_ucy_test"
    elif args.dset_file == "datasets_syn.yaml": # synthetic datasets
        data_file = "synthetic_test"

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

    tb_writer = None

    # (SARL) config path setup
    policy_config_path = "baseline_code/CrowdNav/crowd_nav/configs/policy.config"
    env_config_path = "baseline_code/CrowdNav/crowd_nav/configs/env.config"

    output_dir = "baseline_code/CrowdNav/crowd_nav"
    rl_weight_file = os.path.join(output_dir, 'rl_model.pth')

    # (SARL) configure policy
    import configparser
    from baseline_code.CrowdNav.crowd_nav.policy.sarl import SARL

    policy = SARL()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_path)
    policy.configure(policy_config)
    model_weights = rl_weight_file
    policy.get_model().load_state_dict(torch.load(model_weights))

    policy.set_phase("test")
    policy.set_device(device)
    policy.time_step = sim.dt
    model = policy.get_model()

    # (SARL) read env parameters
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_path)
    # robot_radius = env_config.getfloat('env', 'robot_radius')
    # robot_v_pref = env_config.getfloat('env', 'robot_v_pref') ## use same

    # (SARL) configure explorer
    explorer = Explorer(
        sim, policy, device, gamma=policy.gamma, env_config=env_config,
        mpc_config=mpc_config)

    ########################################################################
    # sim.case_id_list.sort()
    np.random.shuffle(sim.case_id_list)

    obs_data_parser = ObsDataParser(mpc_config, args)

    max_follow_pos_delta = (mpc_config.getint('mpc_env', 'mpc_horizon') *
                            mpc_config.getfloat('mpc_env', 'max_speed'))

    logging.info("-------------------------------------------")
    logging.info(f"Start evaluating SARL policy with {total_eval_episodes} episodes")

    episode = 0
    while episode < total_eval_episodes:
        # (SARL) Sample k episodes into memory
        case_id = explorer.run_k_episodes(1, "test")
        explorer.n_transitions = 0
        episode += 1


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

        ############## save the evaluation results to the csv file ######################
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
        #################################################################################


    # sim = Simulator(args, 'data/eth_0.json', logger)
    # agent = PedNoPredMPC(args, logger)
    # obs = sim.reset(100)
    # done = False
    # while not done:
    #     action = agent.act(obs)
    #     obs, reward, done, info = sim.step(action)
    # sim.evaluate(output=True)
