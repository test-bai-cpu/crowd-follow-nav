import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from os import path

# from gym import spaces
from gymnasium import spaces

from rl.rl_agent import SAC
from rl.replay_buffer import ReplayBuffer


def linear_schedule(n, init=0.2, max_n=100000):
    """
    max_n: Reaches 1 only at the end of learning.
    """
    return min(1., init + (1 - init) * min(n / max_n, 1))



class BaseTrainer(nn.Module):
    def __init__(
        self,
        agent: SAC,
        result_dir,
        config
    ):
        super().__init__()
        self.agent = agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_dir = result_dir
        self.train_steps = 0

        self.config = config
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.train_policy_freq = config["train_policy_freq"]
        self.target_network_update_freq = config["target_network_update_freq"]

        self.n_transitions = 0
        self.global_step = 0

        num_envs = 1
        self.num_envs = num_envs
        self.num_eval_envs = 1
        self.eipsode_rewards = np.zeros(self.num_envs)
        self.eipsode_returns = np.zeros(self.num_envs)
        self.eipsode_lengths = np.zeros(self.num_envs, dtype=int)
        self.eipsode_lengths_eval = np.zeros(self.num_eval_envs, dtype=int)

        # self.buffer_init_beta = config["buffer_init_beta"]
        # self.total_timesteps = config["total_timesteps"]//self.num_envs
        self.training_intensity = config["training_intensity"]
        self.num_eval_episodes = config["num_eval_episodes"]
        self.batch_size = config["batch_size"]

        self.num_samples_before_learning = config["num_samples_before_learning"]//num_envs
        self.train_freq = max(config["train_freq"]//num_envs, 1)
        self.eval_freq = max(config["eval_freq"]//num_envs, 1)
        self.save_model_freq = config["save_model_freq"]//num_envs
        self.report_rollout_freq = config["report_rollout_freq"]//num_envs
        self.report_loss_freq = config["report_loss_freq"]//num_envs

        self.reporter = {'total_reward': [],
                         'total_return': [],
                         'episode_length': [],
                         'hist_complete': []}
            
        self.hist_images = []
        self.obs_shape = config["state_shape"]

        if config.get("autotune", False):
            self.target_entropy = -torch.prod(
                torch.tensor(config["action_shape"]).to(self.device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.target_entropy = None
            entropy_alpha = config.get("entropy_alpha", 0.2)
            log_alpha = torch.tensor(np.log([entropy_alpha]), device=self.device)

        self.alpha = log_alpha.exp().item()
        self.log_alpha = log_alpha

        ##################################################
        # Optimizers
        ##################################################
        q_parameters = list(self.agent.qf1.parameters()) + \
                       list(self.agent.qf2.parameters())
        self.q_optimizer = optim.Adam(
            q_parameters, lr=config["critic_learning_rate"])
        actor_parameters = list(self.agent.actor.parameters())
        self.actor_optimizer = optim.Adam(
            actor_parameters, lr=config["actor_learning_rate"])

        a_optimizer = None
        if config.get("autotune", False):
            self.a_optimizer = optim.Adam(
                [self.log_alpha], lr=config["critic_learning_rate"])

        ##################################################
        # Replay Buffer
        ##################################################
        # A dummy observation and action spaces (no idea how to get from the environment)
        dummy_action_space = spaces.Box(
            low=-1, high=1, shape=config["action_shape"])
        dummy_observation_space = spaces.Box(
            low=-1, high=1, shape=config["state_shape"])

        self.rb = ReplayBuffer(config["buffer_size"], dummy_observation_space,
                               dummy_action_space, self.device,
                               n_envs=self.num_envs)

    def _clip_grad(self):
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(),
                                       self.max_grad_norm)

    ##################################################
    # Info and report progress
    ##################################################
    def update_episode_info(self, rewards):
        self.eipsode_rewards += rewards
        self.eipsode_returns += (rewards * self.gamma**(self.eipsode_lengths))
        self.eipsode_lengths += 1

    def record_episode_info(self, dones, infos):
        reward_list = self.eipsode_rewards[dones]
        return_list = self.eipsode_returns[dones]
        length_list = self.eipsode_lengths[dones]

        self.reporter["total_reward"] += reward_list.tolist()
        self.reporter["total_return"] += return_list.tolist()
        self.reporter["episode_length"] += length_list.tolist()
        if "is_success" in infos:
            self.reporter["hist_complete"] += infos['is_success'][dones].tolist()

        self._reset_episode_info(dones)

    def _reset_episode_info(self, dones):
        self.eipsode_rewards[dones] = 0
        self.eipsode_returns[dones] = 0
        self.eipsode_lengths[dones] = 0

    def report_progress(self, writer, train_info={}):
        if self.is_report_rollout():
            mean_return = np.mean(self.reporter["total_return"])
            mean_reward = np.mean(self.reporter["total_reward"])
            mean_length = np.mean(self.reporter["episode_length"])
            print(f"RL >>> Step: {self.global_step} | Episodic Return: {mean_return:.3f} | Reward: {mean_reward:.3f} | Length: {mean_length}")
            self.reporter["total_return"] = []
            self.reporter["total_reward"] = []
            self.reporter["episode_length"] = []
            # self.report_rollout_info(writer, self.n_transitions)

        # Report loss info
        # if self.is_report_train():
        #     self.report_train_info(writer, train_info)

    ##################################################
    # Conditions
    ##################################################
    def is_learning_starts(self):
        if self.global_step >= self.num_samples_before_learning:
            return True
        return False

    def is_report_rollout(self):
        if self.global_step % self.report_rollout_freq == 0:
            return True
        return False

    def is_train_model(self):
        if (self.is_learning_starts() and
            self.global_step % self.train_freq == 0):
            return True
        return False

    def is_report_train(self):
        if (self.is_learning_starts() and
            self.global_step % self.report_loss_freq == 0):
            return True
        return False

    def is_save_model(self):
        if self.global_step % self.save_model_freq == 0 :
            return True
        return False

    def is_evaluate_model(self):
        if (self.is_learning_starts() and
            self.global_step % self.eval_freq == 0):
            return True
        return False



class ContinuousSACTrainer(BaseTrainer):
    def __init__(
        self,
        agent: SAC,
        result_dir,
        config
    ):
        super().__init__(agent, result_dir, config)

        self.max_grad_norm = 0.5
        self._is_obs_image = False

    def _preprocess(self, data):
        obs = data.observations.to(self.device)
        actions = data.actions.to(self.device)
        next_obs = data.next_observations.to(self.device)
        rewards = data.rewards.flatten()
        dones = data.dones.flatten()
        weights = 1
        return (obs, actions, next_obs, rewards, dones, weights)

    def _gradient_descent(self, optimizer, loss, is_clip_grad=False):
        optimizer.zero_grad()
        loss.backward()

        if is_clip_grad:
            self._clip_grad()
        optimizer.step()

    def add_to_buffer(self, obs, next_obs, actions, rewards, dones, infos):
        self.rb.add(obs, next_obs, actions, rewards, dones, infos)
        self.n_transitions += obs.shape[0]


    ##################################################
    # Training
    ##################################################
    def train_model(self):
        # rb_beta = linear_schedule(self.n_transitions,
        #                           init=self.buffer_init_beta,
        #                           max_n=self.total_timesteps)

        for _ in range(self.training_intensity):
            data = self.rb.sample(self.batch_size)
            train_info = self.train_once(data)
            # self.rb.update_priorities(data.batch_inds, data.env_inds,
            #                      train_info["q_network"]["abs_td_error"].numpy())

        train_info = {**train_info,
                      "n_transitions": self.n_transitions}
        return train_info

    def train_once(self, data):
        """ Main entrence of the training loop. """

        self.train_steps += 1

        # Preprocess data
        obs, actions, next_obs, rewards, dones, weights = \
            self._preprocess(data)

        # Update Q networks
        q_train_info = self._update_q_networks(
            obs, actions, next_obs, rewards, dones, weights)

        # Update actor network
        actor_train_info = {}
        if self.train_steps % self.train_policy_freq == 0:
            actor_train_info = self._update_actor_network(obs, weights)

        # Update the target networks
        if self.train_steps % self.target_network_update_freq == 0:
            self._update_target_newtorks(self.tau)

        return {"reward": rewards.mean().item(),
                "q_network": q_train_info,
                "actor": actor_train_info}

    def save_model(self):
        if not self.is_save_model():
            return

        checkpoint_path = path.join(self.result_dir,
                                    f"n_samples_{self.n_transitions:07d}")
        save_dict = {
            'global_step': self.global_step,
            'num_envs': self.num_envs,
            'num_transitions': self.n_transitions,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': {
                "q_network": self.q_optimizer.state_dict(),
                "actor": self.actor_optimizer.state_dict()},
            "log_alpha": self.log_alpha}
        torch.save(save_dict, checkpoint_path)

        buffer_path = path.join(self.result_dir,
                                f"buffer_{self.n_transitions:08d}.pkl")
        self.rb.save(buffer_path)

        print(f"Save checkpoint in {checkpoint_path}")

    ##############################
    # Update SAC: Q-Networks
    ##############################
    def _compute_target_q(self, next_obs, rewards, dones):
        with torch.no_grad():
            next_actions, next_log_prob, _ = \
                self.agent.get_action(next_obs)

            q1_next_target, q2_next_target = \
                self.agent.get_q_values_target_network(next_obs, next_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            min_q_next_target = (min_q_next_target - self.alpha * next_log_prob).view(-1)

            next_q_value = \
                rewards + (1 - dones) * self.gamma * min_q_next_target
        return next_q_value

    def _update_q_networks(self, obs, actions, next_obs, rewards, dones, weights=None):
        # compute target Q
        next_q_value = self._compute_target_q(next_obs, rewards, dones)

        # Compute Q_t
        q1_values, q2_values = self.agent.get_q_values(obs, actions)
        q1_values, q2_values = q1_values.view(-1), q2_values.view(-1)

        # For prioritized buffer replay (correct bias)
        td_error_1 = next_q_value - q1_values
        td_error_2 = next_q_value - q2_values
        abs_td_error = (torch.abs(td_error_1) + torch.abs(td_error_2)) / 2.
        qf1_loss = torch.mean(td_error_1.pow(2) * weights)
        qf2_loss = torch.mean(td_error_2.pow(2) * weights)
        qf_loss = qf1_loss + qf2_loss

        # Update the model
        self._gradient_descent(self.q_optimizer, qf_loss)

        info = {"q1_values": q1_values.mean().item(),
                "q2_values": q2_values.mean().item(),
                "q1_loss":  qf1_loss.item(),
                "q2_loss": qf2_loss.item(),
                "q_loss": qf_loss.item(),
                "abs_td_error": abs_td_error.cpu().detach()}
        return info

    def _update_target_newtorks(self, tau):
        for param, target_param in zip(self.agent.qf1.parameters(),
                                       self.agent.qf1_target.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

        for param, target_param in zip(self.agent.qf2.parameters(),
                                       self.agent.qf2_target.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    ##############################
    # Update SAC: Actor
    ##############################
    def _autotune_alpha(self, obs):
        with torch.no_grad():
            _, log_pi, _ = self.agent.get_action(obs)

        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

        self._gradient_descent(self.a_optimizer, alpha_loss)

        self.alpha = self.log_alpha.exp().item()
        alpha_loss = alpha_loss.item()
        return alpha_loss

    def _update_actor_network(self, obs, weights=None):
        for _ in range(self.config["train_policy_freq"]):
            actions, log_pi, _ = self.agent.get_action(obs)

            q1_values, q2_values = self.agent.get_q_values(obs, actions)
            min_q_values = torch.min(q1_values, q2_values)

            # Compute loss
            actor_loss = ((self.alpha * log_pi) - min_q_values).mean(dim=1)
            actor_loss = (actor_loss * weights).mean()

            # Update the model
            self._gradient_descent(self.actor_optimizer, actor_loss)

            # Autotune alpha
            alpha_loss = 0
            if self.config.get("autotune", False):
                alpha_loss = self._autotune_alpha(obs)

        info = {"loss": actor_loss.item(),
                "alpha_loss": alpha_loss,
                "alpha": self.alpha}
        return info
