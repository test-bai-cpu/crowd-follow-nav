import numpy as np
import torch
import gym
import os
from time import time

from crowdattn.rl.networks.model import Policy
from sgan.scripts.inference import SGANInference

class CrowdAttnRL(object):
    # The baseline model is the crowd attention RL model
    # Need both crowdattn and sgan to be installed
    # Modified from crowdattn/test.py and crowdattn/rl/evaluate.py

    def __init__(self, args, logger, sgan_model_path, model_dir, ckpt='41665.pt'):
        self.logger = logger
        self.robot_speed = args.robot_speed
        self.predict_steps = args.future_steps
        self.human_num = args.rl_humans

        if not args.no_cuda:
            self.device = torch.device('cuda:' + str(args.gpu_id))
        else:
            self.device = torch.device('cpu')

        self.sgan = SGANInference(sgan_model_path, args.gpu_id)
        self.logger.info('SGAN loaded!')

        from importlib import import_module
        model_dir_temp = model_dir
        if model_dir_temp.endswith('/'):
            model_dir_temp = model_dir_temp[:-1]
        # import arguments.py from saved directory
        # if not found, import from the default directory
        try:
            model_dir_string = model_dir_temp.replace('/', '.') + '.arguments'
            model_arguments = import_module(model_dir_string)
            get_args = getattr(model_arguments, 'get_args')
        except:
            self.logger.warning('Failed to get get_args function from ', model_dir, '/arguments.py')
            from crowdattn.arguments import get_args

        algo_args = get_args()
        torch.manual_seed(algo_args.seed)
        torch.cuda.manual_seed_all(algo_args.seed)
        # if algo_args.cuda:
        #     if algo_args.cuda_deterministic:
        #         # reproducible but slower
        #         torch.backends.cudnn.benchmark = False
        #         torch.backends.cudnn.deterministic = True
        #     else:
        #         # not reproducible but faster
        #         torch.backends.cudnn.benchmark = True
        #         torch.backends.cudnn.deterministic = False
        torch.set_num_threads(1)

        self._set_spaces()

        load_path=os.path.join(model_dir,'checkpoints', ckpt)

        self.actor_critic = Policy(
			self.observation_space.spaces,
			self.action_space,
			base_kwargs=algo_args,
			base='selfAttn_merge_srnn')
        self.actor_critic.load_state_dict(torch.load(load_path, map_location=self.device))
        self.actor_critic.base.nenv = 1
        self.actor_critic.to(self.device)

        self.eval_recurrent_hidden_states = {}

        self.num_processes = 1
        edge_num = self.actor_critic.base.human_num + 1
        self.eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(self.num_processes, 1, self.actor_critic.base.human_node_rnn_size,
                                                                     device=self.device)

        self.eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(self.num_processes, edge_num,
                                                                           self.actor_critic.base.human_human_edge_rnn_size,
                                                                           device=self.device)
        self.eval_masks = torch.zeros(self.num_processes, 1, device=self.device)

        self.logger.info('RL setup complete. Loaded model from {}'.format(load_path))

        self.state_time = []
        self.eval_time = []

        return

    def _set_spaces(self):
        # Modified from crowdattn/crowd_sim/envs/crowd_sim_pred_real_gst.py
        """set observation space and action space"""

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d = {}
        # robot node: num_visible_humans, px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        '''
        format of spatial_edges: [max_human_num, [state_t, state_(t+1), ..., state(t+self.pred_steps)]]
        '''

        # predictions only include mu_x, mu_y (or px, py)
        self.spatial_edge_dim = int(2*(self.predict_steps+1))

        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                            shape=(self.human_num, self.spatial_edge_dim),
                            dtype=np.float32)

        # masks for gst pred model
        # whether each human is visible to robot (ordered by human ID, should not be sorted)
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.human_num,),
                                            dtype=np.bool)

        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        return

    def convert_observation(self, obs):
        # Convert the observation to the format that the RL model expects
        robot_pos = obs['robot_pos']
        robot_goal = obs['robot_goal']
        robot_vel = obs['robot_vel']
        robot_theta = obs['robot_th']

        curr_pos = obs['pedestrians_pos']
        actual_num_ped = len(curr_pos)
        num_ped = min(self.human_num, actual_num_ped)

        if actual_num_ped > 0:
            has_ped = True
            dists = np.linalg.norm(curr_pos - robot_pos, axis=1)
            sorted_idx = np.argsort(dists)
            curr_pos = curr_pos[sorted_idx]
            history_pos = obs['pedestrians_pos_history']

            pos_predictions = self.sgan.evaluate(history_pos)
            pos_predictions = pos_predictions[sorted_idx]
        else:
            has_ped = False
            pos_predictions = []

        obs_rl = {}
        robot_node = torch.zeros((1, 1, 7), dtype=torch.float32, device=self.device)
        robot_node[0, 0, 0] = float(robot_pos[0])
        robot_node[0, 0, 1] = float(robot_pos[1])
        robot_node[0, 0, 2] = 0.3
        robot_node[0, 0, 3] = float(robot_goal[0])
        robot_node[0, 0, 4] = float(robot_goal[1])
        robot_node[0, 0, 5] = self.robot_speed
        robot_node[0, 0, 6] = float(robot_theta)
        obs_rl['robot_node'] = robot_node

        spatial_edges = np.zeros((1, self.human_num, (self.predict_steps + 1) * 2)) + np.inf
        for i in range(num_ped):
            spatial_edges[0, i, 0:2] = curr_pos[i] - robot_pos
            pos_rel_predictions = pos_predictions[i] - robot_pos
            pos_rel_predictions = pos_rel_predictions[:self.predict_steps, :]
            spatial_edges[0, i, 2:] = torch.tensor(pos_rel_predictions.flatten())
        spatial_edges[np.isinf(spatial_edges)] = 15
        obs_rl['spatial_edges'] = torch.tensor(spatial_edges, dtype=torch.float32, device=self.device)

        temporal_edges = torch.zeros((1, 1, 2), dtype=torch.float32, device=self.device)
        temporal_edges[0, 0, 0] = float(robot_vel[0])
        temporal_edges[0, 0, 1] = float(robot_vel[1])
        obs_rl['temporal_edges'] = temporal_edges

        visibile_masks = torch.zeros((1, self.human_num), dtype=torch.float32, device=self.device)
        visibile_masks[0, :num_ped] = 1
        visibile_masks = visibile_masks.bool()
        obs_rl['visible_masks'] = visibile_masks

        detected_human_num = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        detected_human_num[0, 0] = num_ped
        obs_rl['detected_human_num'] = detected_human_num

        return obs_rl, has_ped

    def act(self, obs, done=False):
        # Given an observation, return an action
        state_time_start = time()
        obs_rl, has_ped = self.convert_observation(obs)
        if has_ped == False:
            robot_goal = obs['robot_goal']
            robot_pos = obs['robot_pos']
            robot_speed = self.robot_speed
            action = robot_goal - robot_pos
            act_norm = np.linalg.norm(action)
            action[0] = action[0] / act_norm * robot_speed
            action[1] = action[1] / act_norm * robot_speed
            return action
        state_time_end = time()

        eval_time_start = time()
        with torch.no_grad():
            _, action, _, self.eval_recurrent_hidden_states = self.actor_critic.act(
                obs_rl,
                self.eval_recurrent_hidden_states,
                self.eval_masks,
                deterministic=True)
        action = action.squeeze().cpu().numpy()

        act_norm = np.linalg.norm(action)
        v_pref = self.robot_speed
        if act_norm > v_pref:
            action[0] = action[0] / act_norm * v_pref
            action[1] = action[1] / act_norm * v_pref
        eval_time_end = time()

        done = [done]
        self.eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=self.device)

        self.state_time.append(state_time_end - state_time_start)
        self.eval_time.append(eval_time_end - eval_time_start)
        return action

    def get_processing_time(self):
        # Get processing time for MPC
        if len(self.state_time) == 0 or len(self.eval_time) == 0:
            return None, None
        else:
            return np.mean(self.state_time), np.mean(self.eval_time)
