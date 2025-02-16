import casadi as cs
import configparser
import math
import numpy as np




def dynamics(use_a_omega, differential, state, control, dt):
    if use_a_omega: ### dynamics when (a, omega) are control inputs
        x, y, v, theta = state[0], state[1], state[2], state[3]
        a, omega = control[0], control[1]
    else: ### dynamics when (v, omega) are control inputs
        x, y, theta = state[0], state[1], state[2]
        v, omega = control[0], control[1]

    if differential:
        next_x = cs.if_else(
            cs.fabs(omega) < 1e-6,  # Avoid division by zero
            x + v * cs.cos(theta) * dt,  # Straight motion
            x + (v / omega) * (cs.sin(theta + omega * dt) - cs.sin(theta))  # Circular motion
        )
        next_y = cs.if_else(
            cs.fabs(omega) < 1e-6,
            y + v * cs.sin(theta) * dt,  # Straight motion
            y - (v / omega) * (cs.cos(theta + omega * dt) - cs.cos(theta))  # Circular motion
        )
    else:
        next_x = x + v * cs.cos(theta) * dt
        next_y = y + v * cs.sin(theta) * dt
        
    if use_a_omega: ### dynamics when (a, omega) are control inputs
        next_state = cs.vertcat(
            next_x,
            next_y,
            v + a * dt,
            theta + omega * dt
        )
    else: ### dynamics when (v, omega) are control inputs
        next_state = cs.vertcat(
            next_x,
            next_y,
            theta + omega * dt
        )
        
    return next_state


def parse_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return config


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def circdiff(circular_1, circular_2):
    res = np.arctan2(np.sin(circular_1-circular_2), np.cos(circular_1-circular_2))
    return abs(res)


def distance_wrap_2d(p1, p2):
    ad = circdiff(p1[1], p2[1])
    ld = abs(p1[0] - p2[0])
    dist = math.sqrt(ad**2 + ld**2)
    
    return dist


def distance_wrap_2d_vectorized(A, B):
    ad = circdiff(A[:, 1], B[:, 1])  # Angular differences
    ld = np.abs(A[:, 0] - B[:, 0])  # Linear differences
    dist = np.sqrt(ad**2 + ld**2)
    return dist


def wrapTo2pi(circular_value):
    return np.round(np.mod(circular_value,2*np.pi), 3)


def _circfuncs_common(samples, high, low):
    # Ensure samples are array-like and size is not zero
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = np.sin((samples - low)*2.* np.pi / (high - low))
    cos_samp = np.cos((samples - low)*2.* np.pi / (high - low))

    return samples, sin_samp, cos_samp


def circmean(samples, weights, high=2*np.pi, low=0):
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = sum(sin_samp * weights)
    cos_sum = sum(cos_samp * weights)
    res = np.arctan2(sin_sum, cos_sum)
    res = res*(high - low)/2.0/np.pi + low
    return wrapTo2pi(res)


def circdiff_casadi(theta1, theta2):
    return cs.fabs(cs.atan2(cs.sin(theta1 - theta2), cs.cos(theta1 - theta2)))