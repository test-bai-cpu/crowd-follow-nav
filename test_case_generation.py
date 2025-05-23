import argparse
import os

import json
import numpy as np
from sim.environment import Buffer
from sim.data_loader import DataLoader

# This script is used to generate test cases for the simulator
# Adjust the arguments inside get_args
# and the variables check_regions and start_end_pos
# to generate a new set of test cases
#
# The scrip will go through each dataset with frame intervals of
# interval_factor*fps, and focus on a recrangular region.
# When there are at least num-ppl people inside,
# a test case will be generated assumeing the robot is driving
# on a linear speed of robot_speed from the specified start_end_pos.

def get_args():
    parser = argparse.ArgumentParser(description='test_case_generator')

    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="dt of the simulator"
    )

    parser.add_argument(
        "--robot-speed",
        type=float,
        default=1.75,
        help="suitable robot speed"
    )

    parser.add_argument(
        "--num-ppl",
        type=int,
        default=5,
        help="least number of people in focus region"
    )

    parser.add_argument(
        "--time-factor",
        type=int,
        default=3,
        help="for estimating time limit for each case"
    )

    parser.add_argument(
        "--interval-factor",
        type=int,
        default=3,
        help="do not consider cases within interval_factor*fps"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    
    fps = 1 / args.dt
    datasets = ['eth', 'eth', 'ucy', 'ucy', 'ucy']
    # datasets = ['eth']
    dataset_idxes = [0, 1, 0, 1, 2]
    # dataset_idxes = [0]
    num_datasets = len(datasets)

    robot_speed = args.robot_speed
    least_num_people = args.num_ppl
    time_limit_factor = args.time_factor
    interval_factor = args.interval_factor

    check_start_radius = 1

    # lower-left and upper-right coordinates to specify a square focus region
    check_regions = [[[2, 2], [8, 8]],
                    [[-1, -6], [5, 0]],
                    [[4.5, 2], [10.5, 8]],
                    [[4.5, 3], [10.5, 9]],
                    [[4.5, 3], [10.5, 9]]]
    

    start_end_pos = [[[[-8,5], [15,5]],
                    [[15,5], [-8,5]],
                    [[5,0], [5,12.5]],
                    [[5,12.5], [5,0]]],

                    [[[2,-10.5], [2,4.5]],
                    [[2,4.5], [2,-10.5]],
                    [[5,-3], [-3,-3]],
                    [[-3,-3], [5,-3]]],

                    [[[-0.5,5], [15.5,5]],
                    [[15.5,5], [-0.5,5]],
                    [[7.5,8], [7.5,2]],
                    [[7.5,2], [7.5,8]]],

                    [[[-0.5,6], [16,6]],
                    [[16,6], [-0.5,6]],
                    [[7.5,9], [7.5,2.5]],
                    [[7.5,2.5], [7.5,9]]],

                    [[[0,6], [16,6]],
                    [[16,6], [0,6]],
                    [[7.5,13], [7.5,0]],
                    [[7.5,0], [7.5,13]]]] # num_dset x n x 2 x 2
    
    if not (len(datasets) == len(dataset_idxes) == len(check_regions) == len(start_end_pos)):
        raise Exception("Given dataset information should have the same length!")

    interval = int(interval_factor * fps)
    all_cases = []
    for i in range(num_datasets):
        buffer = Buffer()
        data = DataLoader(datasets[i], dataset_idxes[i], target_fps=fps, base_path="sim")
        buffer = data.update_buffer(buffer)
        pfile_name = os.path.join("data", datasets[i] + "_" + str(dataset_idxes[i]) + '.json')
        cases = []
        
        # check people in focus region
        region = check_regions[i]
        num_frames = len(buffer.video_position_matrix)
        valid_frames = []
        for j in range(0, num_frames, interval):
            pos_array = buffer.video_position_matrix[j]
            num_ped_in_box = 0
            for pos in pos_array:
                if ((pos[0] > region[0][0]) and
                    (pos[0] < region[1][0]) and
                    (pos[1] > region[0][1]) and
                    (pos[1] < region[1][1])):
                    num_ped_in_box += 1
            if num_ped_in_box >= least_num_people:
                valid_frames.append(j)

        # generate test case
        for st_ed in start_end_pos[i]:
            region_mid_pt = np.array([(region[0][0] + region[1][0]) / 2, 
                                    (region[0][1] + region[1][1]) / 2])
            st_pt = np.array(st_ed[0])
            ed_pt = np.array(st_ed[1])
            dist_to_mid = np.linalg.norm(region_mid_pt - st_pt)
            time_to_mid = int(round(dist_to_mid / robot_speed * fps))
            dist_to_end = np.linalg.norm(ed_pt - st_pt)
            time_to_end = int(round(dist_to_end / robot_speed * fps))
            
            count = 0
            time_limit = time_to_end * time_limit_factor
            for v_frame in valid_frames:
                if v_frame >= time_to_mid:
                    start_frame = v_frame - time_to_mid
                    # check if any ped spawn right on top of robot
                    start_frame_ped_pos = buffer.video_position_matrix[start_frame]
                    ped_near_start = False
                    for pos in start_frame_ped_pos:
                        if np.linalg.norm(st_pt - np.array(pos)) < check_start_radius:
                            ped_near_start = True
                            break
                    if not ped_near_start:
                        elem = {'env': datasets[i], 
                                'env_flag': dataset_idxes[i], 
                                'start_pos': st_pt.tolist(), 
                                'end_pos': ed_pt.tolist(), 
                                'start_frame': start_frame, 
                                'time_limit': time_limit}
                        count += 1
                        cases.append(elem)
                        all_cases.append(elem)
            print(count)

        print("Num cases: ", len(cases))
        with open(pfile_name, "w") as fp:
            json.dump(cases, fp)

    with open(os.path.join("data", "all_origin.json"), "w") as fp:
        json.dump(all_cases, fp)

    # # Generate a subset for tunning parameters
    # num_tune = 100
    # # If percentage based, uncomment these
    # # tune_percent = 0.2
    # # num_tune = int(len(all_cases) * tune_percent)
    # tune_cases = []
    # tune_idxes = np.random.permutation(len(all_cases))
    # tune_idxes = tune_idxes[:num_tune]
    # for idx in tune_idxes:
    #     case = all_cases[idx]
    #     tune_cases.append(case)

    # with open(os.path.join("data", "tune.json"), "w") as fp:
    #     json.dump(tune_cases, fp)