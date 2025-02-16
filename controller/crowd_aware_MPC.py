import logging
import time
from copy import deepcopy
import os

import numpy as np
import pandas as pd
import casadi as cs

from matplotlib import pyplot as plt
import pickle

from controller import mpc_utils

class CrowdAwareMPC:
    def __init__(self, config, use_a_omega=False, differential=False):
        if config is not None:
            self.configure(config)
        else:
            raise ValueError('Please provide a configuration file')
        
        self.kinematics = 'kinematic'
        self.use_a_omega = use_a_omega
        self.differential = differential
        
        # mpc solver attributes
        if self.use_a_omega:
            self.nx = 4  # State: [x, y, speed, motion_angle]
            self.nu = 2  # Control: [a, omega]
        else:
            self.nx = 3  # State: [x, y, motion_angle]
            self.nu = 2  # Control: [speed, omega]
        
        # MPC Solver
        self.opti = None
        self.opti_dict = {}
        self.init_solver(ipopt_print_level=0)

    def configure(self, config):
        self.pref_speed =  config.getfloat('mpc_env', 'pref_speed')
        self.max_speed = config.getfloat('mpc_env', 'max_speed')
        self.max_rev_speed = config.getfloat('mpc_env', 'max_speed')
        self.max_rot = config.getfloat('mpc_env', 'max_rot_degrees') * np.pi / 180.0
        self.max_l_acc = config.getfloat('mpc_env', 'max_l_acc')
        self.max_l_dcc = config.getfloat('mpc_env', 'max_l_dcc')
        
        self.max_human_groups = config.getint('mpc_env', 'max_human_groups')
        self.max_humans = config.getint('mpc_env', 'max_humans')
        self.mpc_horizon = config.getint('mpc_env', 'mpc_horizon')
        self.max_mp_steps = config.getint('mpc_env', 'max_mp_steps')
        # self.use_a_omega = config.getboolean('mpc_env', 'use_a_omega')
        # logging.info('[MPCEnv] Config {:} = {:}'.format('pref_speed', self.pref_speed))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_speed', self.max_speed))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_rev_speed', self.max_rev_speed))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_rot', self.max_rot))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_l_acc', self.max_l_acc))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_l_dcc', self.max_l_dcc))
        
        self.w_goal = config.getfloat('mpc_env', 'w_goal')
        self.w_safe = config.getfloat('mpc_env', 'w_safe')
        self.w_follow = config.getfloat('mpc_env', 'w_follow')
        self.w_smooth = config.getfloat('mpc_env', 'w_smooth')
        
        self.d_safe = config.getfloat('mpc_env', 'd_safe')
        
        self.dt = config.getfloat('mpc_env', 'dt')
        
    def init_solver(self, ipopt_print_level=0):
        """Sets up nonlinear optimization problem.
        Adapted from 
        # (safe-control-gym/safe_control_gym/controllers/mpc/mpc.py)
        """

        opti = cs.Opti()
        x_var = opti.variable(self.nx, self.mpc_horizon + 1) # [x, y, speed, motion_angle] / [x, y, motion_angle]
        u_var = opti.variable(self.nu, self.mpc_horizon) # [a, omega] / [speed, omega]
        
        print("################## check x_var shape: ", x_var.shape, "self.nx: ",  self.nx)  
        
        # Initial state and target
        x_init = opti.parameter(self.nx, 1)
        target = opti.parameter(2, 1)
        
        # crowd-related parameters, including all people motion nearby and one following point
        ## nearby human pos and vel is for current time step
        ## follow pos and vel is for all preidciton horizon
        nearby_human_pos = opti.parameter(2, self.max_humans) # [x, y] for all humans
        nearby_human_vel = opti.parameter(2, self.max_humans) # [speed, motion_angle] for all humans
        if_follow = opti.parameter(1, 1) # 0 or 1
        follow_pos = opti.parameter(2, self.mpc_horizon) # [x, y] for all prediction horizon
        follow_vel = opti.parameter(2, self.mpc_horizon) # [speed, motion_angle] for all prediction horizon
        
        # Cost function
        cost = 0
        for t in range(self.mpc_horizon):
            
            # 1. Maintain safe distance cost
            distances = []
            for human_i in range(self.max_humans):
                human_x = nearby_human_pos[0, human_i]
                human_y = nearby_human_pos[1, human_i]
                valid_human = (human_x < 1e6)
                dist = cs.if_else(valid_human, cs.sqrt((x_var[0,t] - human_x)**2 + (x_var[1,t] - human_y)**2), 1e6)
                distances.append(dist)

            min_distance = cs.mmin(cs.vertcat(*distances))
            cost_safe = cs.exp(cs.fmax(self.d_safe - min_distance, 0))
            cost_safe = 0
                
            #Crowd following cost
            if self.use_a_omega:
                cost_follow = if_follow * cs.sumsqr(x_var[:2, t] - follow_pos[:, t])
                ad = mpc_utils.circdiff_casadi(x_var[3, t], follow_vel[1, t])
                ld = cs.fabs(x_var[2, t] - follow_vel[0, t])
                cost_follow += if_follow * cs.sqrt(ad**2 + ld**2)
            else:
                cost_follow = if_follow * cs.sumsqr(x_var[:2, t] - follow_pos[:, t])
                ad = mpc_utils.circdiff_casadi(x_var[2, t], follow_vel[1, t])
                ld = cs.fabs(u_var[0, t] - follow_vel[0, t])
                cost_follow += if_follow * cs.sqrt(ad**2 + ld**2)
        
            # 3. Smooth control
            cost_smooth = 0
            if t > 0:
                cost_smooth = cs.sumsqr(u_var[:, t] - u_var[:, t - 1])

            cost += self.w_safe * cost_safe + self.w_smooth * cost_smooth + self.w_follow * cost_follow
            
        # #Crowd following cost
        # cost_follow = if_follow * cs.sumsqr(x_var[:2, self.mpc_horizon-1] - follow_pos[:, self.mpc_horizon-1])
        # ad = mpc_utils.circdiff_casadi(x_var[3, self.mpc_horizon-1], follow_vel[1, self.mpc_horizon-1])
        # ld = cs.fabs(x_var[2, self.mpc_horizon-1] - follow_vel[0, self.mpc_horizon-1])
        # cost_follow += if_follow * cs.sqrt(ad**2 + ld**2)
        
        cost += self.w_goal * cs.sumsqr(x_var[:2, self.mpc_horizon-1] - target)
        
        opti.minimize(cost)
        
        # Constraints
        for t in range(self.mpc_horizon):
            opti.subject_to(x_var[:, t + 1] == mpc_utils.dynamics(self.use_a_omega, self.differential, x_var[:, t], u_var[:, t], self.dt))
        opti.subject_to(opti.bounded(-self.max_rot, u_var[1, :], self.max_rot))
        
        if self.use_a_omega:
            print("------ use a omega ------")
            opti.subject_to(opti.bounded(-self.max_rev_speed, x_var[2, :], self.max_speed))
            opti.subject_to(opti.bounded(-self.max_l_dcc, u_var[0, :], self.max_l_acc))
        else:
            print("------ use v omega ------")
            opti.subject_to(opti.bounded(-self.max_rev_speed, u_var[0, :], self.max_speed))
        
        # Initial state constraint
        opti.subject_to(x_var[:, 0] == x_init)
        
        # Solver setup
        opts = {
            "ipopt.print_level": ipopt_print_level, 
            "print_time": 0,
            "ipopt.max_iter": self.max_mp_steps,}
        opti.solver("ipopt", opts)
        
        # Store solver attributes
        self.opti = opti
        self.opti_dict = {
            "opti": opti,
            "x_var": x_var,
            "u_var": u_var,
            "x_init": x_init,
            "target": target,
            "nearby_human_pos": nearby_human_pos,
            "nearby_human_vel": nearby_human_vel,
            "if_follow": if_follow,
            "follow_pos": follow_pos,
            "follow_vel": follow_vel,
        }

    def get_action(self, current_state, target, nearby_human_pos, nearby_human_vel, follow_pos, follow_vel):
        """Solves the MPC optimization and returns the next control action."""
        opti = self.opti

        # Set current state
        opti.set_value(self.opti_dict["x_init"], current_state.reshape(-1,1))
        opti.set_value(self.opti_dict["target"], target.reshape(-1,1))

        # Set crowd status
        if nearby_human_pos is not None and nearby_human_vel is not None:
            num_humans = nearby_human_pos.shape[0]
            padded_pos = np.full((self.max_humans, 2), 1e6)
            padded_vel = np.full((self.max_humans, 2), 1e6)
            padded_pos[:num_humans, :] = nearby_human_pos
            padded_vel[:num_humans, :] = nearby_human_vel
            
            opti.set_value(self.opti_dict["nearby_human_pos"], padded_pos.T)
            opti.set_value(self.opti_dict["nearby_human_vel"], padded_vel.T)
        else:
            opti.set_value(self.opti_dict["nearby_human_pos"], np.full((2, self.max_humans), 1e6))
            opti.set_value(self.opti_dict["nearby_human_vel"], np.full((2, self.max_humans), 1e6))
        
        if follow_pos is not None and follow_vel is not None:
            opti.set_value(self.opti_dict["if_follow"], 1)
            opti.set_value(self.opti_dict["follow_pos"], follow_pos.T)
            opti.set_value(self.opti_dict["follow_vel"], follow_vel.T)
        else:
            opti.set_value(self.opti_dict["if_follow"], 0)
            opti.set_value(self.opti_dict["follow_pos"], np.full((2, self.mpc_horizon), 1e6))  # Ignore cost
            opti.set_value(self.opti_dict["follow_vel"], np.full((2, self.mpc_horizon), 1e6))  # Ignore cost


        # Solve the MPC problem
        try:
            sol = opti.solve()
        except:
            print("----------------------------------")
            print("Solver failed. Debugging values:")
            x_var = self.opti_dict["x_var"]
            u_var = self.opti_dict["u_var"]
            print("x_var (state trajectory):", opti.debug.value(x_var))
            print("u_var (control inputs):", opti.debug.value(u_var))
            solver_stats = opti.stats()  # Get solver statistics
            print("Solver return status:", solver_stats["return_status"])
            print("Solver iterations:", solver_stats["iter_count"])
            print("Solver objective value:", solver_stats["iter_stats"]["obj"])
            print("----------------------------------")

            

            raise  # Re-raise error after debugging

        # Extract first control action
        u_opt = sol.value(self.opti_dict["u_var"][:, 0])

        return u_opt, sol.value(self.opti_dict["x_var"])