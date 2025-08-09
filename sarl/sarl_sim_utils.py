import sys, os
import numpy as np
# from baseline_code.CrowdNav.crowd_sim.envs.utils.state import ObservableState
# from baseline_code.CrowdNav.crowd_sim.envs.utils.state import JointState, FullState
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.state import JointState, FullState


def process_obs_to_sarl(origin_obs):
    sarl_obs = []

    ped_positions = origin_obs["pedestrians_pos"].copy()  # shape: (N, 2)
    ped_velocities = origin_obs["pedestrians_vel"].copy()  # shape: (N, 2)

    for pos, vel in zip(ped_positions, ped_velocities):
        px, py = pos
        vx, vy = vel
        ped = ObservableState(px, py, vx, vy, radius=0.25)
        sarl_obs.append(ped)

    return sarl_obs


def get_robot_fullstate(obs, px, py, vx, vy, radius, gx, gy, v_pref, theta):
    full_state = FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta)
    state = JointState(full_state, obs)
    return state


def wrapToPi(angle):
    """
    Wrap an angle in radians to the range [-π, π]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi