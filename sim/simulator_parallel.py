import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
import numpy as np

from obs_data_parser import ObsDataParser
from sim.simulator import Simulator


class SimulatorGym(Simulator, gym.Env):
    def __init__(self, env_config):
        gym.Env.__init__(self)
        Simulator.__init__(self, env_config["args"], env_config["case_fpath"],
                           env_config["logger"])

        self.obs_data_parser = ObsDataParser(env_config["mpc_config"],
                                             env_config["args"])
        self.case_id_idx = 0
        np.random.shuffle(self.case_id_list)
        self.observation_space = spaces.Dict({
            "rl_obs": spaces.Box(
                low=-np.inf, high=np.inf, shape=env_config["observation_shape"],
                dtype=np.float32),
            "current_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=[3,], dtype=np.float32),
            "robot_vy": spaces.Box(
                low=-np.inf, high=np.inf, shape=[1,], dtype=np.float32),
            "robot_vx": spaces.Box(
                low=-np.inf, high=np.inf, shape=[1,],
                dtype=np.float32),
            "target": spaces.Box(
                low=-np.inf, high=np.inf, shape=[2,],
                dtype=np.float32),
            "nearby_human_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=[10, 4], dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-1, high=1, shape=env_config["action_shape"])

    def _preprocess_rl_obs(self, obs, current_state, robot_vx, robot_vy, goal_pos):
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

    def _get_obs(self, observation_dict):
        obs = observation_dict      # To fit the code below
        # These codes were in main.py:146
        current_state, target, robot_speed, robot_motion_angle = self.obs_data_parser.get_robot_state(obs)
        robot_vx = robot_speed * np.cos(robot_motion_angle)
        robot_vy = robot_speed * np.sin(robot_motion_angle)
        nearby_human_state = self.obs_data_parser.get_human_state(obs) ## padding to max_humans, padding with 1e6 (for pos and vel). Human_state is (n, 4): pos_x, pos_y, vel_x, vel_y

        rl_obs = self._preprocess_rl_obs(nearby_human_state, current_state,
                                         robot_vx, robot_vy, self.goal_pos)
        return {
            "rl_obs": rl_obs,
            "current_state": current_state,
            "robot_vy": robot_vy,
            "robot_vx": robot_vx,
            "target": target,
            "nearby_human_state": nearby_human_state}

    def reset(self, seed = None, options = None):
        gym.Env.reset(self, seed=seed)
        case_id = self.case_id_list[self.case_id_idx]
        observation_dict = Simulator.reset(self, case_id)

        obs = self._get_obs(observation_dict)
        # for k, v in obs.items():
        #     print(f"{k}: {v.shape}")
        self.case_id_idx += 1

        self.logger.info(f"Now in the case id: {case_id}")
        return obs, {}

    def step(self, action):
        (observation_dict, reward, done,
         success, time, info_dict) = Simulator.step(self, action)

        obs = self._get_obs(observation_dict)
        terminated = success
        truncated = False if terminated else done

        info = {**info_dict}
        if truncated:
            info["final_observation"] = obs["rl_obs"]
        return obs, reward, terminated, truncated, info
