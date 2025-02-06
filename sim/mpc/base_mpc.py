from abc import ABC, abstractmethod
import numpy as np

class BaseMPC(ABC):
    # Base class for sampling-based MPC

    def __init__(self, args, logger):
        # MPC parameters

        self.laser = args.laser

        self.dt = args.dt
        self.time_horizon = args.time_horizon
        self.future_steps = args.future_steps
        self.big_num_rollouts = args.num_linear
        self.robot_speed = args.robot_speed
        self.logger = logger

        self.dist_weight = 0.5 # weight for distance cost
        self.gamma = 0.9 # discount factor

        self.rollouts = None
        self.num_rollouts = None
        self.rollout_costs = None
        self.robot_pos = None
        self.robot_vel = None
        self.pos_predictions = None
        self.vel_predictions = None
        return

    def generate_rollouts(self):
        # Generate rollouts for MPC
        len_horizon = self.future_steps
        num_rollouts = self.big_num_rollouts
        start_config = self.robot_pos
        dt = self.dt
        vel = self.robot_speed

        angles = np.linspace(np.radians(-180), np.radians(180), num_rollouts, endpoint=True)
        rollouts = np.zeros((num_rollouts * 9 + 1, len_horizon, 2), dtype=np.float32)
        rollouts[:, 0] = start_config
        # radius of curvature, assuming the full curve is a quater circle.
        R1 = vel * dt * (len_horizon - 1) / (np.pi / 2)

        # Generate rollouts
        # Each group of rollouts is generated with three levels of velocities and angular velocities
        # Therefore, the total number of rollouts is 9 times the number of rollouts.
        # Each one is along a different direction.
        # The first group is the fastest, the second group is 2/3 of the fastest, and the third group is 1/3 of the fastest
        # The last one is the stationary rollout.
        for i in range(len_horizon):
            t = i + 1
            rollouts[:num_rollouts, i, 0] = start_config[0] + (vel * dt * t * np.sin(angles[:]))
            rollouts[:num_rollouts, i, 1] = start_config[1] + (vel * dt * t * np.cos(angles[:]))
            rollouts[num_rollouts:(2*num_rollouts), i, 0] = \
                start_config[0] + (2/3 * vel * dt * t * np.sin(angles[:]))
            rollouts[num_rollouts:(2*num_rollouts), i, 1] = \
                start_config[1] + (2/3 * vel * dt * t * np.cos(angles[:]))
            rollouts[(2*num_rollouts):(3*num_rollouts), i, 0] = \
                start_config[0] + (1/3 * vel * dt * t * np.sin(angles[:]))
            rollouts[(2*num_rollouts):(3*num_rollouts), i, 1] = \
                start_config[1] + (1/3 * vel * dt * t * np.cos(angles[:]))

            ang = (vel * dt * t) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(3*num_rollouts):(4*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] + ang))
            rollouts[(3*num_rollouts):(4*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] + ang))
            ang = (2/3 * vel * dt * t) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(4*num_rollouts):(5*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] + ang))
            rollouts[(4*num_rollouts):(5*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] + ang))
            ang = (1/3 * vel * dt * t) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(5*num_rollouts):(6*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] + ang))
            rollouts[(5*num_rollouts):(6*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] + ang))

            ang = (vel * dt * t) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(6*num_rollouts):(7*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] - ang))
            rollouts[(6*num_rollouts):(7*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] - ang))
            ang = (2/3 * vel * dt * t) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(7*num_rollouts):(8*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] - ang))
            rollouts[(7*num_rollouts):(8*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] - ang))
            ang = (1/3 * vel * dt * t) / (2 * R1)
            L = 2 * R1 * np.sin(ang)
            rollouts[(8*num_rollouts):(9*num_rollouts), i, 0] = \
                start_config[0] + (L * np.sin(angles[:] - ang))
            rollouts[(8*num_rollouts):(9*num_rollouts), i, 1] = \
                start_config[1] + (L * np.cos(angles[:] - ang))
            
        rollouts[-1, :, :] = start_config
        self.rollouts = rollouts
        self.num_rollouts =len(rollouts)
        return


    @abstractmethod
    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        pass

    @abstractmethod
    def evaluate_rollouts(self, mpc_weight=None):
        # Evaluate rollouts for MPC
        pass

    @abstractmethod
    def act(self, obs):
        # Get control for MPC, the overall function
        pass
