import numpy as np
from sim.mpc.base_mpc import BaseMPC
from sim.mpc.group import boundary_dist

class PedNoPredMPC(BaseMPC):
    # MPC class for Pedestrian-based representation without prediction
    def __init__(self, args, logger):
        # MPC parameters
        super(PedNoPredMPC, self).__init__(args, logger)
        return

    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        # Repeat the current position for future steps
        curr_pos = obs['pedestrians_pos']
        curr_vel = obs['pedestrians_vel']
        self.boundary_const = obs['personal_size']
        self.pos_predictions = np.repeat(np.expand_dims(curr_pos, axis=1), self.future_steps, axis=1)
        self.vel_predictions = np.repeat(np.expand_dims(curr_vel, axis=1), self.future_steps, axis=1)
        return
    
    def _find_least_dist(self, config, points):
        # Find the least distance between config and the points
        # Inputs:
        # config: the configuration
        # points: the points, dimension is Nx2
        if len(points) == 0:
            return 1e+9, None
        diff = points - config
        dist = np.linalg.norm(diff, axis=1)
        return np.min(dist), np.argmin(dist)
    
    def _rollout_dist(self, rollout, ped_pos, ped_vel, space_const):
        # Calculate the distance between the rollout and the pedestrians
        # Inputs:
        # rollout: the rollout, dimension is Tx2
        # ped_pos: the positions of the pedestrians, dimension is Nx2
        # ped_vel: the velocities of the pedestrians, dimension is Nx2
        # space_const: the constant for calculating the boundary distance
        #
        # Outputs:
        # dists: the distances, dimension is T
        # hit_idx: the index of the time step where the rollout hits the pedestrian

        time_steps = self.future_steps
        dists = np.ones(time_steps) * (1e+9)
        hit_idx = time_steps
        if not (time_steps <= np.shape(ped_pos)[1]):
            self.logger.error('Prediction length is shorter than the time horizon')
            raise ValueError('Prediction length is shorter than the time horizon')
        
        for i in range(time_steps):
            ped_pos_curr = ped_pos[:, i, :]
            ped_vel_curr = ped_vel[:, i, :]
            dists[i], idx = self._find_least_dist(rollout[i], ped_pos_curr)
            if not (idx == None):
                min_ped_vel = ped_vel_curr[idx]
                min_ped_ori = np.arctan2(min_ped_vel[1], min_ped_vel[0])
                rel_pos = rollout[i] - ped_pos_curr[idx]
                rel_ang = np.arctan2(rel_pos[1], rel_pos[0]) - min_ped_ori
                b_dist = boundary_dist(min_ped_vel, rel_ang, space_const)
                if dists[i] <= b_dist:
                    hit_idx = min(hit_idx, i)

        return dists, hit_idx
    
    def _min_dist_cost_func(self, dists, hit_idx):
        cost = 0
        gamma = self.gamma
        discount = 1
        for i, d in enumerate(dists):
            if i >= hit_idx:
                d = -d
            #cost += np.exp(-d)
            cost += np.exp(-d) * discount
            discount *= gamma
        return cost

    def evaluate_rollouts(self, mpc_weight=None):
        # Evaluate rollouts for MPC
        # Rollouts are NxTx2 arrays, where N is the number of rollouts, T is the number of time steps
        # Predictions are MxTx2 arrays, where M is the number of pedesrtians, T is the number of time steps

        if self.rollouts is None or self.pos_predictions is None:
            self.logger.error('Rollouts or predictions are not generated')
            raise ValueError('Rollouts or predictions are not generated')
        
        if mpc_weight is None:
            mpc_weight = self.dist_weight

        if len(self.pos_predictions) == 0:
            has_ped = False
        else:
            has_ped = True

        self.rollout_costs = np.zeros(self.num_rollouts, dtype=np.float32)
        min_dist_weight = mpc_weight
        end_dist_weight = 1 - min_dist_weight

        for i in range(self.num_rollouts):
            # Calculate the distance between the rollouts and predictions
            if has_ped:
                min_dists, hit_idx = self._rollout_dist(self.rollouts[i], 
                                                       self.pos_predictions, 
                                                       self.vel_predictions, 
                                                       self.boundary_const)
                min_dist_cost = self._min_dist_cost_func(min_dists, hit_idx)
            else:
                min_dist_cost = 0
                hit_idx = self.future_steps
            if hit_idx == 0:
                end_dist_cost = np.linalg.norm(self.robot_goal - self.robot_pos)
            else:
                end_dist_cost = np.linalg.norm(self.robot_goal - self.rollouts[i, hit_idx - 1])
            self.rollout_costs[i] = min_dist_weight * min_dist_cost + end_dist_weight * end_dist_cost
        return

    def act(self, obs):
        # Produce action based on observation for the MPC

        self.robot_pos = obs['robot_pos']
        self.robot_goal = obs['robot_goal']
    
        self.generate_rollouts()
        self.get_state_and_predictions(obs)
        self.evaluate_rollouts()
        best_idx = np.argmin(self.rollout_costs)
        action = (self.rollouts[best_idx][0] - self.robot_pos) / self.dt
        return action