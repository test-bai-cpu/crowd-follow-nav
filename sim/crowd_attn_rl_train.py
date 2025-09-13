import numpy as np
import torch
import gym
gym.logger.set_level(gym.logger.ERROR)   # or WARNING
import pickle

import os
import sys
from time import time

from crowdattn.rl import ppo
from crowdattn.rl.networks.model import Policy
from crowdattn.rl.networks.storage import RolloutStorage
from sgan.scripts.inference import SGANInference
from crowdattn.gst_updated.scripts.wrapper.crowd_nav_interface_parallel import CrowdNavPredInterfaceMultiEnv

sys.path.append(os.path.abspath("./crowdattn"))

class CrowdAttnRL_train(object):
    # The baseline model is the crowd attention RL model
    # Need both crowdattn and sgan to be installed
    # Modified from crowdattn/test.py and crowdattn/rl/evaluate.py

    def __init__(self, args, logger, model_dir, ckpt='41665.pt'):
        self.logger = logger
        self.robot_speed = args.robot_speed
        self.predict_steps = args.future_steps
        self.human_num = args.rl_humans
        self.num_processes = 1

        if not args.no_cuda:
            self.device = torch.device('cuda:' + str(args.gpu_id))
        else:
            self.device = torch.device('cpu')

        # self.sgan = SGANInference(sgan_model_path, args.gpu_id)

        load_path = os.path.join(os.getcwd(), "crowdattn/gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj")
        if not os.path.isdir(load_path):
            raise RuntimeError('The result directory was not found.')
        checkpoint_dir = os.path.join(load_path, 'checkpoint')
        with open(os.path.join(checkpoint_dir, 'args.pickle'), 'rb') as f:
            gst_args = pickle.load(f)
        self.predictor = CrowdNavPredInterfaceMultiEnv(
            load_path=load_path, 
            device=self.device, 
            config=gst_args, 
            num_env=1)

        self.logger.info('HMP loaded!')
        
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
        self.algo_args = algo_args
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

        self.actor_critic = Policy(
			self.observation_space.spaces,
			self.action_space,
			base_kwargs=algo_args,
			base='selfAttn_merge_srnn')
        
        self.actor_critic.base.nenv = 1
        self.actor_critic.to(self.device)

        # storage buffer to store the agent's experience
        self.rollouts = RolloutStorage(algo_args.num_steps,
                                1,
                                self.observation_space.spaces,
                                self.action_space,
                                algo_args.human_node_rnn_size,
                                algo_args.human_human_edge_rnn_size)

        # create the ppo optimizer
        self.agent = ppo.PPO(
            self.actor_critic,
            algo_args.clip_param,
            algo_args.ppo_epoch,
            1,
            algo_args.value_loss_coef,
            algo_args.entropy_coef,
            lr=algo_args.lr,
            eps=algo_args.eps,
            max_grad_norm=algo_args.max_grad_norm)

        self.eval_recurrent_hidden_states = {}

        
        edge_num = self.actor_critic.base.human_num + 1
        self.eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(self.num_processes, 1, self.actor_critic.base.human_node_rnn_size,
                                                                     device=self.device)

        self.eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(self.num_processes, edge_num,
                                                                           self.actor_critic.base.human_human_edge_rnn_size,
                                                                           device=self.device)

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
                                            dtype=bool)

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
        # print("goal: ", robot_goal)
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
        
            # pos_predictions = self.sgan.evaluate(history_pos)
            # pos_predictions = pos_predictions[sorted_idx]
            
            in_traj = torch.tensor(history_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
            # take last 5 timesteps
            in_traj = in_traj[:, :, -5:, :]
            in_mask = torch.ones_like(in_traj[..., 0], dtype=torch.float32, device=self.device).unsqueeze(-1)
            out_traj, out_mask = self.predictor.forward(input_traj=in_traj, input_binary_mask=in_mask)
            out_traj = out_traj[:, :, :self.predict_steps, :2].squeeze(0)
            out_traj = out_traj.cpu().numpy()
            pos_predictions = out_traj[sorted_idx]
            
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

    
    def get_processing_time(self):
        # Get processing time for MPC
        if len(self.state_time) == 0 or len(self.eval_time) == 0:
            return None, None
        else:
            return np.mean(self.state_time), np.mean(self.eval_time)
