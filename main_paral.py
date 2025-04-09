import gymnasium as gym
import os, sys
import csv
import logging
import yaml
import numpy as np
import random
import time

from config import get_args, check_args
from sim.simulator import Simulator
from sim.simulator_parallel import SimulatorGym
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
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
    obs = obs.reshape(1, -1)
    obs = np.concatenate([goal_pos, obs], axis=1)
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


def make_env(env_id, seed, kwargs, wrapper_list=[], capture_video=False, run_name="video"):
    def thunk():
        # NOTE: Must set the seed here! (not working outside the make_env.)
        np.random.seed(seed)

        env = gym.make(env_id, **kwargs)

        if capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        for wrapper in wrapper_list:
            env = wrapper(env)
        env.action_space.seed(seed)
        return env

    return thunk


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
    data_file = "all"

    # Load configs
    mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")
    rl_config = load_config("rl_config.yaml")

    # Create simulator
    env_id = "grid_maze_env"
    env_config = {
        "args": args,
        "case_fpath": f"data/{data_file}.json",
        "logger": logger,
        "mpc_config": mpc_config,
        "observation_shape": rl_config["state_shape"],
        "action_shape": rl_config["action_shape"]}

    gym.register(env_id,
                 entry_point="sim.simulator_parallel:SimulatorGym",
                 max_episode_steps=1000,
                 disable_env_checker=True)

    kwargs = {"env_config": env_config}
    # wrapper_list = [gym.wrappers.RecordEpisodeStatistics]
    wrapper_list = []

    env_list = [make_env(env_id, args.seed + i, kwargs, wrapper_list)
                for i in range(args.num_envs)]
    sim = gym.vector.AsyncVectorEnv(env_list)

    output_dir = sim.get_attr("output_dir")[0]
    os.makedirs(os.path.join(output_dir, "evas"), exist_ok=True)
    eva_res_dir = os.path.join(output_dir, "evas", f"{data_file}_{args.exp_name}.csv")
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

    result_dir = os.path.join(rl_config["result_dir"], args.exp_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    rl_agent = SAC(rl_config["state_shape"], rl_config["action_shape"],
                   rl_config["latent_dim"], device)
    rl_trainer = ContinuousSACTrainer(
        rl_agent, result_dir, rl_config, args.num_envs)
    train_info = {}

    tb_writer = SummaryWriter(f"{rl_config['result_dir']}/logs/{args.exp_name}/{int(time.time())}")
    logger.info(f"RL result directory: {result_dir}")
    ########################################################################
    # sim.case_id_list.sort()
    # np.random.shuffle(sim.case_id_list)

    # mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")
    obs_data_parser = ObsDataParser(mpc_config, args)

    # mpc_horizon = mpc_config.getint('mpc_env', 'mpc_horizon')
    # max_speed = mpc_config.getfloat('mpc_env', 'max_speed')
    max_follow_pos_delta = (mpc_config.getint('mpc_env', 'mpc_horizon') *
                            mpc_config.getfloat('mpc_env', 'max_speed'))

    obs, _ = sim.reset()
    mpcs = [CrowdAwareMPC(mpc_config, args.use_a_omega, args.differential)
            for _ in range(args.num_envs)]

    # max_steps = len(sim.get_attr("case_id_list")[0])
    max_steps = 2_000_000
    for _ in range(max_steps):
        # if case_id <= 42:
        #     continue
        # done = False
        # print("step:", _)
        current_state, target, nearby_human_state, robot_vx, robot_vy, rl_obs = \
            obs["current_state"], obs["target"], obs["nearby_human_state"], obs["robot_vx"], obs["robot_vy"], obs["rl_obs"]

        ############ RL model output the follow_pos ############
        rl_trainer.global_step += 1
        if rl_trainer.is_learning_starts():
            rl_actions, _, entropies = rl_agent.get_action(
                torch.FloatTensor(rl_obs).to(device))
            rl_actions = rl_actions.cpu().detach().numpy()
        else:
            rl_actions = rl_agent.random_actions(args.num_envs)

        # Rescale actions. rl_actions is (1, 4): pos_x, pos_y, vel_x, vel_y, and they are all relative values to the robot, both pos and vel
        follow_pos = rl_actions[:, :2].copy()
        follow_vel = rl_actions[:, 2:].copy()

        ## Now rerange the follow_pos and follow_vel
        follow_pos = follow_pos * max_follow_pos_delta     # Since max_follow_pos_delta > 0
        # revert the relative pos to global pos
        follow_pos = follow_pos + current_state[:, :2]

        follow_vel = (follow_vel + 1) * (mpcs[0].max_speed + mpcs[0].max_rev_speed) / 2 - mpcs[0].max_rev_speed     # Since max_rev_speed > 0
        follow_vel = follow_vel + np.concatenate([robot_vx, robot_vy], axis=1)

        follow_speed = np.linalg.norm(follow_vel, axis=1)
        follow_motion_angle = np.mod(np.arctan2(follow_vel[:, 1], follow_vel[:, 0]), 2 * np.pi)

        follow_state = np.concatenate([
            follow_pos, follow_speed.reshape(-1, 1),
            follow_motion_angle.reshape(-1, 1)], axis=1)
        #########################################################

        actions_mpc = []
        for env_idx in range(args.num_envs):
            action_mpc, _ = mpcs[env_idx].get_action(
                current_state[env_idx], target[env_idx],
                nearby_human_state[env_idx], follow_state[env_idx:env_idx+1])
            actions_mpc.append(action_mpc)
        actions_mpc = np.stack(actions_mpc, axis=0)
        next_obs, reward, terminated, truncated, sim_info = sim.step(actions_mpc)

        #################### RL model output the follow_pos #############################
        real_next_rl_obs = next_obs["rl_obs"].copy()
        for idx, truncs in enumerate(truncated):
            if truncs and "final_observation" in sim_info:
                real_next_rl_obs[idx] = sim_info["final_observation"][idx]

        rl_trainer.add_to_buffer(rl_obs, real_next_rl_obs, rl_actions, reward, terminated, [{}])
        rl_trainer.update_episode_info(reward)
        if "reach_goal_reward" in sim_info:
            rl_trainer.reporter["reach_goal_reward"].append(sim_info["reach_goal_reward"])
            rl_trainer.reporter["reach_goal_reward_dense"].append(sim_info["reach_goal_reward_dense"])
            rl_trainer.reporter["group_matching_reward"].append(sim_info["group_matching_reward"])

        if terminated.any() or truncated.any():
            dones = terminated | truncated
            rl_infos = {"is_success": terminated}
            rl_trainer.record_episode_info(dones, rl_infos)
            for env_idx in range(args.num_envs):
                if dones[env_idx]:
                    mpcs[env_idx] = CrowdAwareMPC(mpc_config, args.use_a_omega, args.differential)

        obs = next_obs

        if rl_trainer.is_train_model():
            # print("Training the RL model")
            train_info = rl_trainer.train_model()

        rl_trainer.report_progress(tb_writer, train_info)
        rl_trainer.save_model()

        #################################################################################


        ############## save the evaluation results to the csv file ##############
        if False:
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
