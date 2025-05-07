import numpy as np
from controller.base_mpc import BaseMPC

from controller import mpc_utils

class GroupLinearMPC(BaseMPC):
# MPC class for Group-based representation without prediction

    def __init__(self, mpc_config, args, logger):
        # MPC parameters
        super(GroupLinearMPC, self).__init__(mpc_config, args, logger)

        self.w_goal = mpc_config.getfloat('mpc_env', 'w_goal')
        self.w_safe = mpc_config.getfloat('mpc_env', 'w_safe')
        self.w_follow = mpc_config.getfloat('mpc_env', 'w_follow')
        self.w_smooth = mpc_config.getfloat('mpc_env', 'w_smooth')
        
        self.has_ped = False
        
        return
    
    def linear_predict_location(self, positions, velocities, labels):
        # Linearly predict the future positions and velocities of the groups
        # by using the current central positions and velocities of the groups
        # Inputs:
        # positions: the current positions of the pedestrians Nx2
        # velocities: the current velocities of the pedestrians Nx2
        # labels: the group labels of the pedestrians at the current time N
        # self.mpc_horizon: the length of the prediction sequence
        # dt: the time interval between frames

        # Outputs:
        # future_positions: the predicted future positions of the pedestrians N, self.mpc_horizon, 2

        N = positions.shape[0]
        future_positions = np.zeros((N, self.mpc_horizon, 2))
        all_labels = np.unique(labels)
        label_to_mean_velocity = {}

        for curr_label in all_labels:
            idxes = np.where(labels == curr_label)[0]
            group_velocities = velocities[idxes]
            mean_velocity = np.mean(group_velocities, axis=0)
            label_to_mean_velocity[curr_label] = mean_velocity

        for i in range(N):
            group_vel = label_to_mean_velocity[labels[i]]
            for t in range(self.mpc_horizon):
                future_positions[i, t] = positions[i] + group_vel * self.dt * (t + 1)
                
        return future_positions
    
    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        # Linearly predict the future positions
        if self.laser:
            curr_pos = obs['laser_pos']
            curr_vel = obs['laser_vel']
            group_ids = obs['laser_group_labels']
        else:
            curr_pos = obs['pedestrians_pos']
            curr_vel = obs['pedestrians_vel']
            group_ids = obs['group_labels']

        num_ped = len(curr_pos)

        if not num_ped == 0:
            self.has_ped = True
            self.future_positions = self.linear_predict_location(curr_pos, curr_vel, group_ids)
        else:
            self.has_ped = False
            self.future_positions = None
            
        return
    
    
    def _rollout_dist(self, rollout, ped_future_positions):
        # Calculate the distance between the rollouts and predictions
        # ped_future_positions, N pedestrians, self.mpc_horizon, 2
        # rollout: self.mpc_horizon, 2
        time_steps = np.shape(rollout)[0]
        dists = np.ones(time_steps)*(1e+9)

        for i in range(time_steps):
            dists[i] = self._get_least_dist(rollout[i], ped_future_positions[:, i])

        return dists
    
    def _get_least_dist(self, rollout, ped_future_positions):
        # ped_future_positions, N pedestrians, 2
        # rollout: 2
        dists = np.linalg.norm(ped_future_positions - rollout, axis=1)  # shape: (N,)
        min_dist = np.min(dists)
        
        return min_dist
        
    
    def _min_dist_cost_func(self, dists):
        cost = 0
        gamma = self.gamma
        discount = 1
        for i, d in enumerate(dists):
            cost += np.exp(self.collision_radius - d) * discount
            # print("check collision radius: ",self.collision_radius)
            discount *= gamma
        return cost

    def evaluate_rollouts(self):
        # Evaluate rollouts for MPC
        # Rollouts are NxTx2 arrays, where N is the number of rollouts, T is the number of time steps
        # Predictions are an array of frames and an array of group boundaries coordinates
        # The array of frames is TxHxW, where H is the height and W is the width
        # The array of group boundaries is a list of TxNx2 arrays, where N is the number of groups

        if self.rollouts is None:
            self.logger.error('Rollouts are not generated')
            raise ValueError('Rollouts are not generated')

        self.rollout_costs = np.zeros(self.num_rollouts, dtype=np.float32)

        for i in range(self.num_rollouts):
            # Calculate the distance between the rollouts and predictions
            if self.has_ped:
                min_dists = self._rollout_dist(self.rollouts[i], self.future_positions)
                cost_safe = self._min_dist_cost_func(min_dists)
            else:
                cost_safe = 0

            # self.rollouts_action shape: self.mpc_horizon, 2
            # action_diffs = np.diff(self.rollouts_action[i], axis=0)
            # cost_smooth = np.sum(np.square(action_diffs))
            
            # follow_state: np.array([follow_pos[0], follow_pos[1], follow_speed, follow_motion_angle])
            follow_pos = self.follow_state[0, :2]

            # follow_vel = self.follow_state[0, 2:]
            # position_cost = np.linalg.norm(self.rollouts[i, self.mpc_horizon-1] - follow_pos)
            position_cost = np.min(np.linalg.norm(self.rollouts[i] - follow_pos, axis=1))
            
            # current_theta = self.robot_th + self.rollouts_action[i, 1] * self.dt * self.mpc_horizon
            # current_theta = np.mod(current_theta,2*np.pi)
            # ad = mpc_utils.circdiff(current_theta, follow_vel[1])
            # ld = np.abs(self.rollouts_action[i, 0] - follow_vel[0])
            # velocity_cost = np.sqrt(ld**2 + ad**2)
            # cost_follow = position_cost + velocity_cost
            cost_follow = position_cost
            # print("-----")
            # print("self.rollouts_action: ", self.rollouts_action[i])
            # print("cost follow: ", cost_follow)
            # print("cost safe: ", cost_safe)
            
            self.rollout_costs[i] = self.w_safe * cost_safe + self.w_follow * cost_follow
            # self.rollout_costs[i] = cost_follow
            
        return
