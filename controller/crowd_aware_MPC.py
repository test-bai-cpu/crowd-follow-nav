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
        self.max_rot = config.getfloat('mpc_env', 'max_rot_degrees')
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
        
        # Initial state and target
        x_init = opti.parameter(self.nx, 1)
        target = opti.parameter(2, 1)
        
        # crowd-related parameters, including all people motion nearby and one following point
        ## nearby human pos and vel is for current time step
        ## follow pos and vel is for all preidciton horizon
        nearby_human_pos = opti.parameter(2, self.max_humans) # [x, y] for all humans
        nearby_human_vel = opti.parameter(2, self.max_humans) # [vx, vy] for all humans
        if_follow = opti.parameter(1, 1) # 0 or 1
        follow_pos = opti.parameter(2, 1) # [x, y] for all prediction horizon. Set to 1 now
        follow_vel = opti.parameter(2, 1) # [speed, motion_angle] for all prediction horizon. Set to 1 now
        
        # Cost function
        cost_safe = 0
        cost_smooth = 0

        for t in range(self.mpc_horizon):
            # 1. Maintain safe distance cost
            distances = []
            for human_i in range(self.max_humans):
                human_x = nearby_human_pos[0, human_i]
                human_y = nearby_human_pos[1, human_i]
                valid_human = (human_x < 1e6)
                # dist = cs.if_else(valid_human, cs.sqrt((x_var[0,t] - human_x)**2 + (x_var[1,t] - human_y)**2), 1e6)
                dist = cs.if_else(valid_human, (x_var[0,t] - human_x)**2 + (x_var[1,t] - human_y)**2, 1e6)
                distances.append(dist)

            # # min_distance = cs.mmin(cs.vertcat(*distances))
            distances = cs.vertcat(*distances)
            has_valid_human = cs.sum1(cs.if_else(distances < 1e6, 1, 0)) > 0

            # Softmin to approximate minimum distance
            alpha = 5  # Lower values make it closer to true min
            softmin_distance = cs.sum1(distances * cs.exp(-distances / alpha)) / cs.sum1(cs.exp(-distances / alpha))

            # If no valid humans, assign large safe distance
            min_distance = cs.if_else(has_valid_human, softmin_distance, 1e6)

            # Compute safety cost (only when close to humans)
            cost_safe += cs.fmax(0, (self.d_safe - min_distance)**2) * self.w_safe
                
            # #Crowd following cost, along all the time_horizon
            # if self.use_a_omega:
            #     cost_follow = if_follow * cs.sumsqr(x_var[:2, t] - follow_pos[:, t])
            #     ad = mpc_utils.circdiff_casadi(x_var[3, t], follow_vel[1, t])
            #     ld = cs.fabs(x_var[2, t] - follow_vel[0, t])
            #     cost_follow += if_follow * cs.sqrt(ad**2 + ld**2)
            # else:
            #     cost_follow = if_follow * cs.sumsqr(x_var[:2, t] - follow_pos[:, t])
            #     ad = mpc_utils.circdiff_casadi(x_var[2, t], follow_vel[1, t])
            #     ld = cs.fabs(u_var[0, t] - follow_vel[0, t])
            #     cost_follow += if_follow * cs.sqrt(ad**2 + ld**2)
        
            # 3. Smooth control
            if t > 0:
                cost_smooth += cs.sumsqr(u_var[:, t] - u_var[:, t - 1]) * self.w_smooth
            
        #Crowd following cost, only check the last time step
        if self.use_a_omega:
            ad = mpc_utils.circdiff_casadi(x_var[3, self.mpc_horizon-1], follow_vel[1, 0])
            ld = (x_var[2, self.mpc_horizon-1] - follow_vel[0, 0])
        else:
            ad = mpc_utils.circdiff_casadi(x_var[2, self.mpc_horizon-1], follow_vel[1, 0])
            ld = (u_var[0, self.mpc_horizon-1] - follow_vel[0, 0])
            
        # cost_follow = self.w_follow * (cs.sumsqr(x_var[:2, self.mpc_horizon-1] - follow_pos[:, 0]) + cs.sqrt(ad**2 + ld**2))
        # cost_follow = self.w_follow * (cs.sumsqr(x_var[:2, self.mpc_horizon-1] - follow_pos[:, 0])) 
        cost_follow = self.w_follow * (cs.sumsqr(x_var[:2, self.mpc_horizon-1] - follow_pos[:, 0]) + ad**2 + ld**2)        
        cost_goal = self.w_goal * cs.sumsqr(x_var[:2, self.mpc_horizon-1] - target)
        
        # cost += self.w_safe * cost_safe + self.w_smooth * cost_smooth
        
        
        ##### TODO: it can happen that with the safety cost, the robot cannot move straightly to the goal, even goal is very
        ##### close. Maybe it because there is person suddently comes from the edge?
        # cost = cost_goal * (1-if_follow) + cost_follow * if_follow + cost_safe + cost_smooth
        # cost = cost_goal + cost_follow + cost_safe + cost_smooth
        cost = cost_follow + cost_safe + cost_smooth
        
        opti.minimize(cost)
        
        # Constraints
        for t in range(self.mpc_horizon):
            opti.subject_to(x_var[:, t + 1] == mpc_utils.dynamics(self.use_a_omega, self.differential, x_var[:, t], u_var[:, t], self.dt))
        opti.subject_to(opti.bounded(-self.max_rot, u_var[1, :], self.max_rot))
        
        if self.use_a_omega:
            # print("------ use a omega ------")
            opti.subject_to(opti.bounded(-self.max_rev_speed, x_var[2, :], self.max_speed))
            opti.subject_to(opti.bounded(-self.max_l_dcc, u_var[0, :], self.max_l_acc))
        else:
            # print("------ use v omega ------")
            opti.subject_to(opti.bounded(-self.max_rev_speed, u_var[0, :], self.max_speed))
        
        # Initial state constraint
        opti.subject_to(x_var[:, 0] == x_init)
        
        # Solver setup
        # opts = {
        #     "ipopt.print_level": ipopt_print_level, 
        #     "print_time": 0,
        #     "ipopt.max_iter": self.max_mp_steps,}
        
        opts = {
            "ipopt.print_level": ipopt_print_level,
            "print_time": 0,
            "ipopt.max_iter": 5000,  # Adjustable based on performance
            "ipopt.acceptable_iter": 10,  # Stop if at least 10 "acceptable" iterations are found
            "ipopt.acceptable_tol": 1e-2,  # Allow a relaxed tolerance
            "ipopt.acceptable_obj_change_tol": 1e-2  # Stop if improvement is small
        }
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

    def get_action(self, current_state, target, nearby_human_state, follow_state):
        """Solves the MPC optimization and returns the next control action."""
        ## follow_state is in speed, motion_angle
        ## nearby_human_state is in pos_x, pos_y, vel_x, vel_y, which padding to max_humans
        opti = self.opti

        # Set current state
        opti.set_value(self.opti_dict["x_init"], current_state.reshape(-1,1))
        opti.set_value(self.opti_dict["target"], target.reshape(-1,1))

        # Set crowd status
        nearby_human_pos = nearby_human_state[:, :2]
        nearby_human_vel = nearby_human_state[:, 2:]
        opti.set_value(self.opti_dict["nearby_human_pos"], nearby_human_pos.T)
        opti.set_value(self.opti_dict["nearby_human_vel"], nearby_human_vel.T)
        
        follow_pos = follow_state[:, :2]
        follow_vel = follow_state[:, 2:]
        opti.set_value(self.opti_dict["follow_pos"], follow_pos.T)
        opti.set_value(self.opti_dict["follow_vel"], follow_vel.T)
        
        ## check if follow_state is the padded value
        if_follow = 0
        if np.sum(follow_vel) < 1e6:
            if_follow = 1
        opti.set_value(self.opti_dict["if_follow"], if_follow)

        # Solve the MPC problem
        try:
            sol = opti.solve()
        except:
            # print("----------------------------------")
            # print("Solver failed. Debugging values:")
            # x_var = opti.debug.value(self.opti_dict["x_var"])
            # u_var = opti.debug.value(self.opti_dict["u_var"])
            # print("x_var (state trajectory):", x_var)
            # print("u_var (control inputs):", u_var)
            
            # follow_vel = opti.debug.value(self.opti_dict["follow_vel"])
            # follow_pos = opti.debug.value(self.opti_dict["follow_pos"])
            
            # print("follow_vel: ", follow_vel)
            # print("follow_pos: ", follow_pos)
            
            # nearby_human_pos = opti.debug.value(self.opti_dict["nearby_human_pos"])
            # nearby_human_vel = opti.debug.value(self.opti_dict["nearby_human_vel"])
            
            # cost_safe = self.get_cost_safe(x_var, nearby_human_pos)
            # cost_smooth = self.get_cost_smooth(u_var)
            # cost_follow = self.get_cost_follow(x_var, u_var, follow_pos, follow_vel, if_follow)
            # cost_goal = self.get_cost_goal(x_var, target)

            # print("cost_safe: ", cost_safe)
            # print("cost_smooth: ", cost_smooth)
            # print("cost_follow: ", cost_follow)
            # print("cost_goal: ", cost_goal)
            
            # solver_stats = opti.stats()  # Get solver statistics
            # print("Solver return status:", solver_stats["return_status"])
            # print("Solver iterations:", solver_stats["iter_count"])
            # print("Solver objective value:", solver_stats["iter_stats"]["obj"])
            # print("----------------------------------")

            # raise  # Re-raise error after debugging
            
            ## set a random action
            print("MPC Solver failed. Set a random action")
            if self.use_a_omega:
                a = 0
                w = 0
                u_opt = np.array([a, w])
            else:
                v = 0
                w = 0
                u_opt = np.array([v, w])
            return u_opt, current_state
        
        ################ Extract costs after solving #########################
        # x_var = sol.value(self.opti_dict["x_var"])
        # u_var = sol.value(self.opti_dict["u_var"])
        # cost = sol.value(self.opti_dict["opti"].f)
        # follow_vel = sol.value(self.opti_dict["follow_vel"])
        # follow_pos = sol.value(self.opti_dict["follow_pos"])
        # if_follow = sol.value(self.opti_dict["if_follow"])
        # nearby_human_pos = sol.value(self.opti_dict["nearby_human_pos"])
        # nearby_human_vel = sol.value(self.opti_dict["nearby_human_vel"])
        
        # cost_safe = self.get_cost_safe(x_var, nearby_human_pos)
        # cost_smooth = self.get_cost_smooth(u_var)
        # cost_follow = self.get_cost_follow(x_var, u_var, follow_pos, follow_vel, if_follow)
        # cost_goal = self.get_cost_goal(x_var, target)
        
        # print("cost_safe: ", cost_safe)
        # print("cost_smooth: ", cost_smooth)
        # print("cost_follow: ", cost_follow)
        # print("cost_goal: ", cost_goal)
        # print("total cost: ", cost)
        
        # print("follow weight: ", self.w_follow, " goal weight: ", self.w_goal)
        # print("if_follow: ", sol.value(if_follow))
        ######################################################################
        
        # Extract first control action
        u_opt = sol.value(self.opti_dict["u_var"][:, 0])

        return u_opt, sol.value(self.opti_dict["x_var"])
    

    def get_cost_safe(self, x_var, nearby_human_pos):
        cost_safe = 0
        for t in range(self.mpc_horizon):
            # 1. Maintain safe distance cost
            distances = []
            for human_i in range(self.max_humans):
                human_x = nearby_human_pos[0, human_i]
                human_y = nearby_human_pos[1, human_i]
                valid_human = (human_x < 1e6)
                if human_x < 1e6:
                    dist = cs.if_else(valid_human, (x_var[0,t] - human_x)**2 + (x_var[1,t] - human_y)**2, 1e6)
                    distances.append(dist)

            distances = cs.vertcat(*distances)
            has_valid_human = cs.sum1(cs.if_else(distances < 1e6, 1, 0)) > 0

            # Softmin to approximate minimum distance
            alpha = 5  # Lower values make it closer to true min
            softmin_distance = cs.sum1(distances * cs.exp(-distances / alpha)) / cs.sum1(cs.exp(-distances / alpha))

            # If no valid humans, assign large safe distance
            min_distance = cs.if_else(has_valid_human, softmin_distance, 1e6)

            # Compute safety cost (only when close to humans)
            cost_safe += cs.fmax(0, (self.d_safe - min_distance)**2) * self.w_safe
            
        return cost_safe
    
    def get_cost_smooth(self, u_var):
        cost_smooth = 0
        for t in range(self.mpc_horizon):
            cost_smooth += cs.sumsqr(u_var[:, t] - u_var[:, t - 1])
            
        return cost_smooth * self.w_smooth
    
    def get_cost_follow(self, x_var, u_var, follow_pos, follow_vel, if_follow):
        if self.use_a_omega:
            ad = mpc_utils.circdiff_casadi(x_var[3, self.mpc_horizon-1], follow_vel[1])
            ld = (x_var[2, self.mpc_horizon-1] - follow_vel[0])
        else:
            ad = mpc_utils.circdiff_casadi(x_var[2, self.mpc_horizon-1], follow_vel[1])
            ld = (u_var[0, self.mpc_horizon-1] - follow_vel[0])
            
        # cost_follow = self.w_follow * (cs.sumsqr(x_var[:2, self.mpc_horizon-1] - follow_pos[:, 0]) + cs.sqrt(ad**2 + ld**2))
        # cost_follow = self.w_follow * (cs.sumsqr(x_var[:2, self.mpc_horizon-1] - follow_pos[:, 0])) 
        cost_follow = self.w_follow * (cs.sumsqr(x_var[:2, self.mpc_horizon-1] - follow_pos) + ad**2 + ld**2)
        cost_follow = if_follow * cost_follow
        
        return cost_follow
    
    def get_cost_goal(self, x_var, target):
        cost_goal = self.w_goal * cs.sumsqr(x_var[:2, self.mpc_horizon-1] - target)
        
        return cost_goal