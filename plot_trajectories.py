#!/usr/bin/env python3
"""
Script to read and plot trajectories from saved pickle files.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import argparse
import os

def load_trajectory_data(filepath):
    """
    Load trajectory data from pickle file.
    
    Args:
        filepath (str): Path to the pickle file
        
    Returns:
        dict: Dictionary containing trajectory data for each case_id
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_single_case_trajectory(data, case_id, save_plot=False, output_dir=None):
    """
    Plot trajectory for a single case.
    
    Args:
        data (dict): Trajectory data dictionary
        case_id (int): Case ID to plot
        save_plot (bool): Whether to save the plot
        output_dir (str): Directory to save plots
    """
    if case_id not in data:
        print(f"Case {case_id} not found in data")
        return
    
    trajectory = data[case_id]
    
    # Extract robot trajectory
    robot_positions = np.array([entry['robot_pos'] for entry in trajectory])
    
    # Extract pedestrian trajectories
    pedestrian_positions = []
    for entry in trajectory:
        if len(entry['pedestrians_pos']) > 0:
            pedestrian_positions.append(entry['pedestrians_pos'])
        else:
            pedestrian_positions.append(np.empty((0, 2)))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot robot trajectory as a line
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax.scatter(robot_positions[0, 0], robot_positions[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(robot_positions[-1, 0], robot_positions[-1, 1], c='red', s=100, marker='x', label='End')
    
    # Plot pedestrian positions as scatter points (not connected lines)
    if len(pedestrian_positions) > 0 and len(pedestrian_positions[0]) > 0:
        num_pedestrians = pedestrian_positions[0].shape[0]
        
        for ped_idx in range(num_pedestrians):
            ped_positions = []
            for t in range(len(pedestrian_positions)):
                if len(pedestrian_positions[t]) > ped_idx:
                    ped_positions.append(pedestrian_positions[t][ped_idx])
                else:
                    ped_positions.append([np.nan, np.nan])
            
            ped_positions = np.array(ped_positions)
            valid_mask = ~np.isnan(ped_positions[:, 0])
            
            if np.any(valid_mask):
                valid_positions = ped_positions[valid_mask]
                # Plot as scatter points instead of connected lines
                ax.scatter(valid_positions[:, 0], valid_positions[:, 1], 
                          color='red', s=30, alpha=0.6, 
                          label=f'Pedestrian {ped_idx+1}' if ped_idx == 0 else "")
                
                # Mark start and end points
                if len(valid_positions) > 0:
                    ax.scatter(valid_positions[0, 0], valid_positions[0, 1], 
                             color='red', s=100, marker='s', alpha=0.8, edgecolors='black')
                    ax.scatter(valid_positions[-1, 0], valid_positions[-1, 1], 
                             color='red', s=100, marker='^', alpha=0.8, edgecolors='black')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Trajectories for Case {case_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'case_{case_id}_trajectory.png'), dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(output_dir, f'case_{case_id}_trajectory.png')}")
    
    plt.show()

def create_animation(data, case_id, save_animation=False, output_dir=None):
    """
    Create an animation of the trajectories.
    
    Args:
        data (dict): Trajectory data dictionary
        case_id (int): Case ID to animate
        save_animation (bool): Whether to save the animation
        output_dir (str): Directory to save animations
    """
    if case_id not in data:
        print(f"Case {case_id} not found in data")
        return
    
    trajectory = data[case_id]
    
    # Extract robot trajectory
    robot_positions = np.array([entry['robot_pos'] for entry in trajectory])
    
    # Extract pedestrian trajectories
    pedestrian_positions = []
    for entry in trajectory:
        if len(entry['pedestrians_pos']) > 0:
            pedestrian_positions.append(entry['pedestrians_pos'])
        else:
            pedestrian_positions.append(np.empty((0, 2)))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up the plot
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Trajectory Animation for Case {case_id}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set plot limits
    all_x = [robot_positions[:, 0]]
    all_y = [robot_positions[:, 1]]
    
    if len(pedestrian_positions) > 0 and len(pedestrian_positions[0]) > 0:
        for ped_traj in pedestrian_positions:
            if len(ped_traj) > 0:
                all_x.append(ped_traj[:, 0])
                all_y.append(ped_traj[:, 1])
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    margin = 2.0
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Initialize plot elements
    robot_line, = ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
    robot_point = ax.scatter([], [], color='blue', s=100, marker='o', label='Robot')
    
    # Pedestrian elements - as scatter points, not lines
    ped_points = []
    
    for i in range(max(1, len(pedestrian_positions[0]) if len(pedestrian_positions) > 0 else 1)):
        point = ax.scatter([], [], color='red', s=50, marker='o', alpha=0.7, label=f'Pedestrian {i+1}' if i == 0 else "")
        ped_points.append(point)
    
    ax.legend()
    
    def animate(frame):
        # Update robot trajectory
        robot_line.set_data(robot_positions[:frame+1, 0], robot_positions[:frame+1, 1])
        robot_point.set_offsets([robot_positions[frame, 0], robot_positions[frame, 1]])
        
        # Update pedestrian positions as scatter points
        if len(pedestrian_positions) > 0 and len(pedestrian_positions[0]) > 0:
            for ped_idx in range(len(pedestrian_positions[0])):
                if frame < len(pedestrian_positions) and len(pedestrian_positions[frame]) > ped_idx:
                    ped_points[ped_idx].set_offsets([pedestrian_positions[frame][ped_idx, 0], 
                                                   pedestrian_positions[frame][ped_idx, 1]])
                else:
                    ped_points[ped_idx].set_offsets([np.nan, np.nan])
        
        return [robot_line, robot_point] + ped_points
    
    anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
                                 interval=100, blit=True, repeat=True)
    
    if save_animation and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        anim.save(os.path.join(output_dir, f'case_{case_id}_animation.gif'), 
                 writer='pillow', fps=10)
        print(f"Animation saved to {os.path.join(output_dir, f'case_{case_id}_animation.gif')}")
    
    plt.show()
    return anim

def plot_all_cases_overview(data, save_plot=False, output_dir=None):
    """
    Create an overview plot showing all cases.
    
    Args:
        data (dict): Trajectory data dictionary
        save_plot (bool): Whether to save the plot
        output_dir (str): Directory to save plots
    """
    num_cases = len(data)
    cols = min(4, num_cases)
    rows = (num_cases + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if num_cases == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for idx, (case_id, trajectory) in enumerate(data.items()):
        ax = axes[idx]
        
        # Extract robot trajectory
        robot_positions = np.array([entry['robot_pos'] for entry in trajectory])
        
        # Plot robot trajectory as line
        ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=2)
        ax.scatter(robot_positions[0, 0], robot_positions[0, 1], color='green', s=50, marker='o')
        ax.scatter(robot_positions[-1, 0], robot_positions[-1, 1], color='red', s=50, marker='x')
        
        # Plot pedestrian positions as scatter points
        pedestrian_positions = []
        for entry in trajectory:
            if len(entry['pedestrians_pos']) > 0:
                pedestrian_positions.append(entry['pedestrians_pos'])
            else:
                pedestrian_positions.append(np.empty((0, 2)))
        
        if len(pedestrian_positions) > 0 and len(pedestrian_positions[0]) > 0:
            num_pedestrians = pedestrian_positions[0].shape[0]
            
            for ped_idx in range(num_pedestrians):
                ped_positions = []
                for t in range(len(pedestrian_positions)):
                    if len(pedestrian_positions[t]) > ped_idx:
                        ped_positions.append(pedestrian_positions[t][ped_idx])
                    else:
                        ped_positions.append([np.nan, np.nan])
                
                ped_positions = np.array(ped_positions)
                valid_mask = ~np.isnan(ped_positions[:, 0])
                
                if np.any(valid_mask):
                    valid_positions = ped_positions[valid_mask]
                    # Plot as scatter points instead of connected lines
                    ax.scatter(valid_positions[:, 0], valid_positions[:, 1], 
                             color='red', alpha=0.6, s=20)
        
        ax.set_title(f'Case {case_id}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide empty subplots
    for idx in range(num_cases, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'all_cases_overview.png'), dpi=300, bbox_inches='tight')
        print(f"Overview plot saved to {os.path.join(output_dir, 'all_cases_overview.png')}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot trajectories from pickle files')
    parser.add_argument('--filepath', type=str, 
                       default='exps/results_all/e012_rw100_1_0_react_sfmrobot_c0.5_h9_syn/evas/synthetic_test_e012_rw100_1_0_react_sfmrobot_c0.5_h9_syn.pkl',
                       help='Path to the pickle file')
    parser.add_argument('--case_id', type=int, default=None, 
                       help='Specific case ID to plot (if None, plot all)')
    parser.add_argument('--animation', action='store_true', 
                       help='Create animation instead of static plot')
    parser.add_argument('--save', action='store_true', 
                       help='Save plots/animations')
    parser.add_argument('--output_dir', type=str, default='trajectory_plots', 
                       help='Output directory for saved plots')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading trajectory data from {args.filepath}")
    data = load_trajectory_data(args.filepath)
    print(f"Loaded {len(data)} cases: {list(data.keys())}")
    
    # Plot data
    if args.case_id is not None:
        if args.animation:
            create_animation(data, args.case_id, args.save, args.output_dir)
        else:
            plot_single_case_trajectory(data, args.case_id, args.save, args.output_dir)
    else:
        plot_all_cases_overview(data, args.save, args.output_dir)

if __name__ == "__main__":
    main() 