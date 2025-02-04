import numpy as np
from sim.mpc.base_mpc import BaseMPC

class PedNoPredMPC(BaseMPC):
    # MPC class for Pedestrian simulator without prediction
    def __init__(self, args):
        # MPC parameters
        super(PedNoPredMPC, self).__init__(args)
        return

    def get_predictions(self, obs):
        # Get predictions for MPC
        return

    def evaluate_rollouts(self):
        # Evaluate rollouts for MPC
        return

    def get_control(self):
        # Get control for MPC
        return np.array([1, 0])