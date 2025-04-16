import numpy as np



def get_start_end_region(dataset, dataset_idx):
    start_region = []
    goal_region = []
    
    if dataset == 'eth' and dataset_idx == 0: ### walking in horizontal
        start_region = [[[-8, -2], [-2, 10]], [[14, 2.5], [17, 7.5]]]
        goal_region = [[[14, 2.5], [17, 7.5]], [[-8, -2], [-2, 10]]]
        # start_region = [[[-8, -2], [-2, 10]]]
        # goal_region = [[[14, 2.5], [17, 7.5]]]
        
    elif dataset == 'eth' and dataset_idx == 1: ## UCY ### walking in vertical
        start_region = [[2, -12], [5, -9]], [[-1, 3], [2, 6]]
        goal_region = [[2, 3], [5, 6]], [[-1, -12], [2, -9]]
        
    elif dataset == 'ucy' and dataset_idx in [0,1]: ## Zara
        start_region = [[[-1.5, 2], [1.5, 10]], [[13.5, 2], [16.5, 10]], [[11.5, 11], [14.5, 13]]]
        goal_region = [[[13.5, 2], [16.5, 10]], [[-1.5, 2], [1.5, 10]], [[-1.5, 2], [1.5, 10]]]
        
    elif dataset == 'ucy' and dataset_idx == 2: ## Univ
        start_region = [[[-1.5, -1], [1.5, 6]], [[14, -1], [16, 9]]]
        goal_region = [[[14, -1], [16, 9]], [[-1.5, -1], [1.5, 6]]]
        
    return start_region, goal_region


def get_start_end_loc(start_region_list, goal_region_list, sample_num=1):
    if len(start_region_list) != len(goal_region_list):
        raise ValueError("start_region_list and goal_region_list must have the same length")

    start_end_loc_list = []
    
    for i in range(len(start_region_list)):
        start_region = start_region_list[i]
        goal_region = goal_region_list[i]

        
        for _ in range(sample_num):
            # Sample from start region
            start_x = np.random.uniform(start_region[0][0], start_region[1][0])
            start_y = np.random.uniform(start_region[0][1], start_region[1][1])
            start_point = [start_x, start_y]

            # Sample from goal region
            goal_x = np.random.uniform(goal_region[0][0], goal_region[1][0])
            goal_y = np.random.uniform(goal_region[0][1], goal_region[1][1])
            goal_point = [goal_x, goal_y]

            start_end_loc_list.append([start_point, goal_point])

    return start_end_loc_list

def get_start_end_loc_with_dataset(dataset, dataset_idx, sample_num=1):
    start_region, goal_region = get_start_end_region(dataset, dataset_idx)
    start_end_loc_list = get_start_end_loc(start_region, goal_region, sample_num)
    
    return start_end_loc_list