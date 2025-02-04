from abc import ABC, abstractmethod
import numpy as np

class BaseMPC(ABC):
    # Base class for sampling-based MPC

    def __init__(self, args):
        # MPC parameters

        self.dt = args.dt
        self.time_horizon = args.time_horizon
        self.future_steps = args.future_steps
        self.num_rollouts = args.num_rollouts
        self.robot_speed = args.robot_speed

        self.rollouts = None
        self.rollout_costs = None
        return

    def generate_rollouts(self, start_config):
        # Generate rollouts for MPC
        len_horizon = self.future_steps
        num_rollouts = self.num_rollouts
        dt = self.dt
        vel = self.robot_speed

        angles = np.linspace(np.radians(-180), np.radians(180), num_rollouts, endpoint=True)
        rollouts = np.zeros((num_rollouts * 9 + 1, num_rollouts, 2), dtype=np.float32)
        rollouts[:, 0] = start_config
        R1 = vel * dt * (len_horizon - 1) / (np.pi / 2)

        # Generate rollouts
        # Each group of rollouts is generated with three levels of velocities and angular velocities
        # Therefore, the total number of rollouts is 9 times the number of rollouts.
        # Each one is along a different direction.
        # The first group is the fastest, the second group is 2/3 of the fastest, and the third group is 1/3 of the fastest
        # The last one is the stationary rollout.
        for i in range(1, len_horizon):
            rollouts[:num_rollouts, i, 0] = start_config[0] + (vel * dt * i * np.sin(angles[:]))
            rollouts[:num_rollouts, i, 1] = start_config[1] + (vel * dt * i * np.cos(angles[:]))
            rollouts[num_rollouts:(2*num_rollouts), i, 0] = \
                start_config[0] + (2/3 * vel * dt * i * np.sin(angles[:]))
            rollouts[num_rollouts:(2*num_rollouts), i, 1] = \
                start_config[1] + (2/3 * vel * dt * i * np.cos(angles[:]))
            rollouts[(2*num_rollouts):(3*num_rollouts), i, 0] = \
                start_config[0] + (1/3 * vel * dt * i * np.sin(angles[:]))
            rollouts[(2*num_rollouts):(3*num_rollouts), i, 1] = \
                start_config[1] + (1/3 * vel * dt * i * np.cos(angles[:]))

            ang = (vel * dt * i) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(3*num_rollouts):(4*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] + ang))
            rollouts[(3*num_rollouts):(4*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] + ang))
            ang = (2/3 * vel * dt * i) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(4*num_rollouts):(5*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] + ang))
            rollouts[(4*num_rollouts):(5*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] + ang))
            ang = (1/3 * vel * dt * i) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(5*num_rollouts):(6*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] + ang))
            rollouts[(5*num_rollouts):(6*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] + ang))

            ang = (vel * dt * i) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(6*num_rollouts):(7*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] - ang))
            rollouts[(6*num_rollouts):(7*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] - ang))
            ang = (2/3 * vel * dt * i) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(7*num_rollouts):(8*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] - ang))
            rollouts[(7*num_rollouts):(8*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] - ang))
            ang = (1/3 * vel * dt * i) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(8*num_rollouts):, i, 0] = \
                start_config[0] + (L * np.sin(angles[:] - ang))
            rollouts[(8*num_rollouts):, i, 1] = \
                start_config[1] + (L * np.cos(angles[:] - ang))
            
        rollouts[-1, :, :] = start_config
        self.rollouts = rollouts
        return


    @abstractmethod
    def get_predictions(self, obs):
        # Get predictions for MPC
        pass

    @abstractmethod
    def evaluate_rollouts(self):
        # Evaluate rollouts for MPC
        pass

    @abstractmethod
    def get_control(self):
        # Get control for MPC
        pass
