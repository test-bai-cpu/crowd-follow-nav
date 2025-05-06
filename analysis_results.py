import numpy as np
import pandas as pd

def get_results(res_file):
    df = pd.read_csv(res_file)

    # Compute success rate
    total_cases = len(df)
    print("In exp: ", res_file)
    print("total cases: ", total_cases)
    successful_cases = df['success'].sum()  # True is treated as 1
    success_rate = successful_cases / total_cases
    
    fail_cases = df[df['success'] == 0]
    num_fail_cases = len(fail_cases)
    
    ### get how many cases are failed due to collision
    collision_cases = fail_cases[fail_cases['fail_reason'] == 'Collision']
    num_collision_cases = len(collision_cases)
    
    ### get how many cases are failed due to timeout
    timeout_cases = fail_cases[fail_cases['fail_reason'] == 'Time']
    num_timeout_cases = len(timeout_cases)

    navigation_time_mean = df['navigation_time'].mean()
    navigation_time_std = df['navigation_time'].std()

    path_length_mean = df['path_length'].mean()
    path_length_std = df['path_length'].std()

    smoothness_mean = df["path_smoothness"].mean()
    smoothness_std = df["path_smoothness"].std()

    motion_smoothness_mean = df["motion_smoothness"].mean()
    motion_smoothness_std = df["motion_smoothness"].std()

    min_ped_dist_mean = df["min_ped_dist"].mean()
    min_ped_dist_std = df["min_ped_dist"].std()

    avg_ped_dist_mean = df["avg_ped_dist"].mean()
    avg_ped_dist_std = df["avg_ped_dist"].std()

    # # Print results
    # print(f"Success rate: {success_rate:.2%}")
    # print(f"Navigation time: {navigation_time_mean:.2f} ± {navigation_time_std:.2f} seconds")
    # print(f"Path length: {path_length_mean:.2f} ± {path_length_std:.2f} meters")
    # print(f"Path smoothness: {smoothness_mean:.4f} ± {smoothness_std:.4f} meters")
    # print(f"Motion smoothness: {motion_smoothness_mean:.4f} ± {motion_smoothness_std:.4f} meters")
    # print(f"Min pedestrian distance: {min_ped_dist_mean:.2f} ± {min_ped_dist_std:.2f} meters")
    # print(f"Avg pedestrian distance: {avg_ped_dist_mean:.2f} ± {avg_ped_dist_std:.2f} meters")

    exp_name  = res_file.split("/")[-1].split(".")[0]
    
    result = {
        "experiment": exp_name,
        "success_rate": success_rate,
        "num_fail_cases": num_fail_cases,
        "num_collision_cases": num_collision_cases,
        "num_timeout_cases": num_timeout_cases,
        "navigation_time_mean": navigation_time_mean,
        "navigation_time_std": navigation_time_std,
        "path_length_mean": path_length_mean,
        "path_length_std": path_length_std,
        "path_smoothness_mean": smoothness_mean,
        "path_smoothness_std": smoothness_std,
        "motion_smoothness_mean": motion_smoothness_mean,
        "motion_smoothness_std": motion_smoothness_std,
        "min_ped_dist_mean": min_ped_dist_mean,
        "min_ped_dist_std": min_ped_dist_std,
        "avg_ped_dist_mean": avg_ped_dist_mean,
        "avg_ped_dist_std": avg_ped_dist_std,
    }
    
    return result
    
files = [
    # "exps/results/evas/test_e005_all_mpcSafeCost_rlReward_fw5.csv",
    # "exps/results/evas/test_e005_all_mpcSafeCost_rlReward.csv",
    # "exps/results/evas/test_e005_all_mpcSafeCost.csv"
    # "exps/results/evas/test_e007_rw100_1_1_react_sfmnorobot_noreward.csv",
    # "exps/results/evas/test_e007_rw100_1_1_react_sfmnorobot.csv"
    # "exps/183/all_origin_e007_rw100_1_1_react_sfmnorobot_noreward.csv",
    # "exps/183/all_origin_e007_rw100_1_1_react_sfmnorobot.csv"
    # "exps/results/evas/synthetic_train2_e008_rw100_1_1_react_sfmrobot_reward2.csv",
    # "exps/results/evas/synthetic_train2_e008_rw100_1_1_react_sfmrobot_noreward2.csv",
    # "exps/results/evas/synthetic_train2_e008_rw100_1_1_react_sfmrobot_reward2_w5.csv"
    # "exps/results_synthetic/synthetic_test2_e009_mpc_linear.csv"
    # "exps/results/evas/synthetic_train3_e010_rw100_1_0_react_sfmrobot_c0.5_h3.csv",
    # "exps/results/evas/synthetic_train3_e010_rw100_1_1_react_sfmrobot_c0.5_h3.csv",
    # "exps/results/evas/synthetic_train3_e010_rw100_1_5_react_sfmrobot_c0.5_h3.csv"
    
    # "exps/results_ethucy/e010_rw100_1_5_react_sfmrobot_c0.1_h5/evas/test_e010_rw100_1_5_react_sfmrobot_c0.1_h5.csv",
    # "exps/results/evas/test_e005_all_mpcSafeCost_rlReward_fw5.csv",
    # "exps/results/evas/test_e005_all_mpcSafeCost_rlReward.csv",
    # "exps/results/evas/test_e005_all_mpcSafeCost.csv"
    
    "exps/results_ethucy/e010_rw100_1_0_react_sfmrobot_c0.1_h5/evas/test_e010_rw100_1_0_react_sfmrobot_c0.1_h5.csv",
    "exps/results_ethucy/e010_rw100_1_5_react_sfmrobot_c0.1_h4/evas/test_e010_rw100_1_5_react_sfmrobot_c0.1_h4.csv",
    "exps/results_ethucy/e010_rw100_1_5_react_sfmrobot_c0.1_h5/evas/test_e010_rw100_1_5_react_sfmrobot_c0.1_h5.csv"
]

all_results = [get_results(f) for f in files]

results_df = pd.DataFrame(all_results)
results_df.to_csv("exps/results_ethucy/summary_results_eth_3.csv", index=False)

# get_results("exps/results/evas/test_e005_all_mpcSafeCost_rlReward_fw5.csv")
# get_results("exps/results/evas/test_e005_all_mpcSafeCost_rlReward.csv")
# get_results("exps/results/evas/test_e005_all_mpcSafeCost.csv")