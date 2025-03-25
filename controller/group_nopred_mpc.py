import numpy as np
import copy
import matplotlib.pyplot as plt

from scipy import signal
from sim.mpc.base_mpc import BaseMPC
from sim.mpc.group import draw_all_social_spaces
from sim.mpc.img_process import DrawGroupShape

class GroupNoPredMPC(BaseMPC):
# MPC class for Group-based representation without prediction

    def __init__(self, args, logger, dataset_info):
        # MPC parameters
        super(GroupNoPredMPC, self).__init__(args, logger)
        if self.laser:
            self.offset = args.ped_size
        else:
            self.offset = 0

        self.dataset_info = dataset_info
        self.frame_predictions = None
        self.boundary_predictions = None
        return

    def _get_frame(self, dataset_info, positions, velocities, group_ids, personal_size):
        # Get the frame for the current time step
        # Draw the group shapes on the frame
        group_vertices = draw_all_social_spaces(group_ids, positions, velocities, personal_size, self.offset)
        canvas = np.zeros((dataset_info['frame_height'], dataset_info['frame_width'], 3), dtype=np.uint8)

        dgs = DrawGroupShape(dataset_info)
        for v in group_vertices:
            canvas = dgs.draw_group_shape(v, canvas, center=False, aug=False)
        img = canvas[:, :, 0] / 255.0

        return img

    def _inv_coordinate_transform(self, dataset_info, vertices):
        # Inverse coordinate transformation
        # Convert vertices to frame coordinates

        if dataset_info['dataset'] == 'ucy':
            tmp = copy.deepcopy(vertices[0,:])
            vertices[0,:] = vertices[1, :] - dataset_info['frame_width'] / 2
            vertices[1,:] = dataset_info['frame_height'] / 2 - tmp
        vertices = np.append(vertices, np.ones((1, np.shape(vertices)[1])), axis=0)
        vertices = np.matmul(dataset_info['H'], vertices)
        vertices = [vertices[0,:] / vertices[2,:], vertices[1,:] / vertices[2,:]]
        return vertices

    def _frame_to_vertices(self, dataset_info, frame):
        # Convert frame to vertices 

        laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        frame = signal.convolve2d(frame, laplacian, mode='same')
        frame = np.clip(np.abs(frame), 0, 1)
        vertices = np.array(np.nonzero(frame))
        vertices = self._inv_coordinate_transform(dataset_info, vertices)

        return np.transpose(np.array(vertices), (1,0))
    
    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        # Repeat the current position for future steps
        if self.laser:
            curr_pos = obs['laser_pos']
            curr_vel = obs['laser_vel']
            group_ids = obs['laser_group_labels']
        else:
            curr_pos = obs['pedestrians_pos']
            curr_vel = obs['pedestrians_vel']
            group_ids = obs['group_labels']
        self.boundary_const = obs['personal_size']

        if not len(group_ids) == 0:
            frame = self._get_frame(self.dataset_info, curr_pos, curr_vel, group_ids, self.boundary_const)
            group_boundary = self._frame_to_vertices(self.dataset_info, frame)

            self.frame_predictions = [frame] * self.future_steps
            self.boundary_predictions = [group_boundary] * self.future_steps

            if self.animate and (not self.paint_boundary):
                self.boundary_pts = draw_all_social_spaces(group_ids, curr_pos, curr_vel, self.boundary_const, self.offset)
        else:
            self.frame_predictions = []
            self.boundary_predictions = []

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
        
    def _check_inside_groups(self, dataset_info, rollout_pt, group_frame):
        # Check if the rollout point is inside the group
        dgs = DrawGroupShape(dataset_info)
        rollout_pt_pix = dgs.coordinate_transform(rollout_pt)
        y, x = rollout_pt_pix
        if ((x >= 0) and (x < dataset_info['frame_height']) and
            (y >= 0) and (y < dataset_info['frame_width']) and
            (group_frame[x, y] > 0)):
            return True
        else:
            return False
    
    def _rollout_dist(self, dataset_info, rollout, group_frames, groups_boundaries):
        # Calculate the distance between the rollouts and predictions
        time_steps = np.shape(rollout)[0]
        dists = np.ones(time_steps)*(1e+9)
        hit_idx = time_steps

        if not (len(group_frames) == len(groups_boundaries)):
            self.logger.error('Group frames and boundaries do not match')
            raise ValueError('Group frames and boundaries do not match')

        if not (time_steps <= len(group_frames)):
            self.logger.error('Prediction length is shorter than the time horizon')
            raise ValueError('Prediction length is shorter than the time horizon')
            
        for i in range(time_steps):
            if self._check_inside_groups(dataset_info, rollout[i], group_frames[i]):
                hit_idx = min(hit_idx, i)
            dists[i], _ = self._find_least_dist(rollout[i], groups_boundaries[i])
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
        # Predictions are an array of frames and an array of group boundaries coordinates
        # The array of frames is TxHxW, where H is the height and W is the width
        # The array of group boundaries is a list of TxNx2 arrays, where N is the number of groups

        if self.rollouts is None or self.frame_predictions is None:
            self.logger.error('Rollouts or predictions are not generated')
            raise ValueError('Rollouts or predictions are not generated')
        
        if self.dataset_info is None:
            self.logger.error('Dataset information is not set')
            raise ValueError('Dataset information is not set')
        
        if mpc_weight is None:
            mpc_weight = self.dist_weight

        if len(self.frame_predictions ) == 0:
            has_ped = False
        else:
            has_ped = True

        self.rollout_costs = np.zeros(self.num_rollouts, dtype=np.float32)
        min_dist_weight = mpc_weight
        end_dist_weight = 1 - min_dist_weight # currently only 2 cost terms

        for i in range(self.num_rollouts):
            # Calculate the distance between the rollouts and predictions
            if has_ped:
                min_dists, hit_idx = self._rollout_dist(self.dataset_info,
                                                        self.rollouts[i], 
                                                        self.frame_predictions, 
                                                        self.boundary_predictions)
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
    
    def add_boundaries(self, frame):
        # Add the boundaries to the frame for rendering
        # This is outside the simulator, so paint-boundary need to stay off

        if self.boundary_pts is None:
            self.logger.error('Boundary points are not set')
            raise ValueError('Boundary points are not set')
        
        for boundary in self.boundary_pts:
            boundary.append(boundary[0])
            boundary_pts = np.array(boundary)
            boundary, = plt.plot(boundary_pts[:, 0], boundary_pts[:, 1], c='k', linewidth=1)
            frame.append(boundary)
        return frame