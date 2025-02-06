import numpy as np
from sim.mpc.base_mpc import BaseMPC

class PedNoPredMPC(BaseMPC):
    # MPC class for Pedestrian-based representation without prediction
    def __init__(self, args):
        # MPC parameters
        super(PedNoPredMPC, self).__init__(args)
        return

    def get_predictions(self, obs):
        # Get predictions for MPC
        curr_pos = obs['pedestrians_pos']
        self.pos_predictions = np.repeat(np.expand_dims(curr_pos, axis=1), self.future_steps, axis=1)
        return

    def evaluate_rollouts(self):
        # Evaluate rollouts for MPC
        return

    def get_control(self):
        # Get control for MPC
        return np.array([1, 0])