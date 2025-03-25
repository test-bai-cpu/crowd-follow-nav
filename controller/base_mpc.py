from abc import ABC, abstractmethod
from time import time
import numpy as np

class BaseMPC(ABC):
    # Base class for sampling-based MPC

    def __init__(self, args, logger):
        # MPC parameters

        self.animate = args.animate
        self.paint_boundary = args.paint_boundary
        if self.animate and (not self.paint_boundary):
            self.boundary_pts = None

        self.laser = args.laser
        self.differential = args.differential

        self.dt = args.dt
        self.time_horizon = args.time_horizon
        self.future_steps = args.future_steps
        self.num_directions = args.num_directions
        self.num_linear = args.num_linear
        self.num_angular = args.num_angular
        self.robot_speed = args.robot_speed
        self.turn_speed = args.turn_speed
        self.logger = logger

        self.dist_weight = args.dist_weight # weight for distance cost
        self.goal_weight = args.goal_weight
        self.gamma = args.gamma # discount factor

        self.rollouts = None
        self.rollouts_action = None
        self.num_rollouts = None
        self.rollout_costs = None
        self.robot_pos = None
        self.robot_th = None
        self.robot_vel = None

        self.state_time = []
        self.eval_time = []
        return

    def generate_rollouts(self):
        # Generate rollouts for MPC
        len_horizon = self.future_steps
        
        if self.robot_pos is None:
            self.logger.error('Robot position is not set')
            raise ValueError('Robot position is not set')
        start_config = self.robot_pos
        dt = self.dt
        vel = self.robot_speed
        omg = self.turn_speed

        if self.differential:
            if self.robot_th is None:
                self.logger.error('Robot orientation is not set')
                raise ValueError('Robot orientation is not set')
            start_th = self.robot_th
            linear_vels = np.linspace(0, vel, self.num_linear)
            linear_vels = linear_vels[1:]
            angular_vels = np.linspace(0, omg, int((self.num_angular + 1) / 2))
            angular_vels = np.concatenate((angular_vels, -angular_vels[1:]))

            # Create all (v, w) pairs using meshgrid.
            V_grid, W_grid = np.meshgrid(linear_vels, angular_vels, indexing='ij')
            V = V_grid.flatten()  # shape: (num_rollouts,)
            W = W_grid.flatten()  # shape: (num_rollouts,)
            V = np.append(V, 0)
            W = np.append(W, 0)
            num_rollouts = V.shape[0]
            rollouts = np.zeros((num_rollouts, len_horizon, 2), dtype=np.float32)

            epsilon = 1e-6
            cond = (np.abs(W) < epsilon)
            W = np.where(cond, epsilon, W)

            # Generate rollouts for each (v, w) pair
            for i in range(len_horizon):
                t = (i + 1) * dt
                rollouts[:, i, 0] = start_config[0] + np.where(cond, V * t * np.cos(start_th), 
                                                               V / W * (np.sin(start_th + W * t) - np.sin(start_th)))
                rollouts[:, i, 1] = start_config[1] + np.where(cond, V * t * np.sin(start_th),
                                                               -V / W * (np.cos(start_th + W * t) - np.cos(start_th)))

            rollouts_action = np.stack((V, W), axis=-1)

        else:
            num_rollouts = self.num_directions

            angles = np.linspace(np.radians(-180), np.radians(180), num_rollouts, endpoint=True)
            rollouts = np.zeros((num_rollouts * 9 + 1, len_horizon, 2), dtype=np.float32)
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
            rollouts_action = (rollouts[:, 0, :] - start_config) / dt

        self.rollouts = rollouts
        self.rollouts_action = rollouts_action
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
    def add_boundaries(self, frame):
        # Add boundaries to the frame for viaulizing boundaries
        pass
        
    def get_processing_time(self):
        # Get processing time for MPC
        if len(self.state_time) == 0 or len(self.eval_time) == 0:
            return None, None
        else:
            return np.mean(self.state_time), np.mean(self.eval_time)

    def act(self, obs, done=False):
        # Produce action based on observation for the MPC
        # Inputs:
        # obs: the observation
        # Outputs:
        # action: the action
        # time: the time spent on state and evaluation

        self.robot_pos = obs['robot_pos']
        self.robot_vel = obs['robot_vel']
        self.robot_th = obs['robot_th']
        self.robot_goal = obs['robot_goal']
    
        self.generate_rollouts()

        state_time_start = time()
        self.get_state_and_predictions(obs)
        state_time_end = time()

        eval_time_start = time()
        self.evaluate_rollouts()
        eval_time_end = time()
        
        best_idx = np.argmin(self.rollout_costs)
        action = self.rollouts_action[best_idx]

        self.state_time.append(state_time_end - state_time_start)
        self.eval_time.append(eval_time_end - eval_time_start)

        return action