import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class SAC(nn.Module):
    def __init__(self, state_shape, action_shape, latent_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        self.state_dim = np.prod(state_shape)
        self.action_dim = np.prod(action_shape)

        self._build_policy(self.state_dim, action_shape, latent_dim, device)
        self._build_q_networks(self.state_dim, action_shape, latent_dim, device)

    def _build_policy(self, state_dim, action_shape, latent_dim, device):
        self.actor = ContinuousActor(
            state_dim, action_shape, latent_dim).to(device)

    def _build_q_networks(self, state_dim, action_shape, latent_dim, device):
        self.qf1 = ContinuousSoftQNetwork(
            state_dim, action_shape, latent_dim).to(device)
        self.qf2 = ContinuousSoftQNetwork(
            state_dim, action_shape, latent_dim).to(device)

        self.qf1_target = ContinuousSoftQNetwork(
            state_dim, action_shape, latent_dim).to(device)
        self.qf2_target = ContinuousSoftQNetwork(
            state_dim, action_shape, latent_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        for param in self.qf1_target.parameters():
            param.requires_grad = False
        for param in self.qf2_target.parameters():
            param.requires_grad = False

    def random_actions(self, num=1):
        actions = np.random.rand(num, self.action_dim)
        return actions

    def get_action(self, x, with_exploration=True):
        if with_exploration:
            action, log_prob, entropy = self.actor.get_action(x)
            return action, log_prob, entropy
        else:
            action = self.actor.get_optimal_action(x)
            return action

    def get_q_values(self, x, a):
        qf1 = self.qf1.forward(x, a)
        qf2 = self.qf2.forward(x, a)
        return qf1, qf2

    def get_q_values_target_network(self, x, a):
        qf1_target = self.qf1_target.forward(x, a)
        qf2_target = self.qf2_target.forward(x, a)
        return qf1_target, qf2_target

    def load_pretrained_agent(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Load SAC agent from: {path_to_checkpoint}")


class ContinuousSoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_shape, latent_dim=128):
        super().__init__()
        action_shape = np.prod(action_shape)

        _dim = latent_dim
        self.v_head = nn.Sequential(
            nn.Linear(state_dim + action_shape, _dim),
            nn.ReLU(),
            nn.Linear(_dim, _dim),
            nn.ReLU(),
            nn.Linear(_dim, 1))

    def forward(self, x, a):
        v = self.v_head(torch.cat([x, a], dim=1))
        return v


class ContinuousActor(nn.ContinuousSoftQNetwork):
    def __init__(self, state_dim, action_shape, latent_dim=64, action_range=[-1, 1]):
        super().__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20  # -5

        action_shape = np.prod(action_shape)

        _dim = latent_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(state_dim, _dim), nn.ReLU(),
            nn.Linear(_dim, _dim), nn.ReLU())

        self.fc_mean = nn.Linear(_dim, action_shape)
        self.fc_logstd = nn.Linear(_dim, action_shape)

        # Put in state_dict but not model parametes
        self.register_buffer(
            "action_scale", torch.tensor(
                (action_range[1] - action_range[0]) / 2.0,
                dtype=torch.float32))
        self.register_buffer(
            "action_bias", torch.tensor(
                (action_range[1] + action_range[0]) / 2.0,
                dtype=torch.float32))

    def forward(self, x):
        features = self.fc_layers(x)
        mean = self.fc_mean(features)
        log_std = self.fc_logstd(features)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + \
                  0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        policy_dist = Normal(mean, std)

        # Reparameterization trick: mean + std * N(0,1)
        x_t = policy_dist.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = policy_dist.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, policy_dist.entropy().sum(1)

    def get_optimal_action(self, x):
        mean, _ = self.forward(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean
