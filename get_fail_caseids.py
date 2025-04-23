import pandas as pd

# case_id,start_frame,success,fail_reason,navigation_time,path_length,path_smoothness,motion_smoothness,min_ped_dist,avg_ped_dist,min_laser_dist,avg_laser_dist
### Get the fail cases, which fail for noreward, but succesff for reward
def get_fail_cases_fornoreward():
    # Load both result files
    df_noreward = pd.read_csv("exps/results/evas/test_e007_rw100_1_1_react_sfmnorobot_noreward.csv")
    df_rlreward = pd.read_csv("exps/results/evas/test_e007_rw100_1_1_react_sfmnorobot.csv")

    # Merge on case_id to align rows
    merged = pd.merge(df_noreward, df_rlreward, on="case_id", suffixes=("_noreward", "_rlreward"))

    # Find cases that failed in noreward but succeeded in rlReward
    recovered_cases = merged[(merged['success_noreward'] == 0) & (merged['success_rlreward'] == 1)]

    # Extract relevant info: case_id and fail_reason from noreward
    result_df = recovered_cases[['case_id', 'fail_reason_noreward']]
    result_df.rename(columns={'fail_reason_noreward': 'fail_reason'})

    # Save to CSV
    result_df.to_csv("exps/failed_cases.csv", index=False)

    print("Saved recovered cases to exps/failed_cases_2.csv")


# get_fail_cases_fornoreward()


def get_fail_cases(res_file, output_file):
    df = pd.read_csv(res_file)

    # Find cases that failed in noreward but succeeded in rlReward
    recovered_cases = df[df['success'] == 0]

    # Extract relevant info: case_id and fail_reason from noreward
    result_df = recovered_cases[['case_id', 'fail_reason']]

    # Save to CSV
    result_df.to_csv(output_file, index=False)
    

# res_file = "exps/results/evas/test_e007_rw100_1_1_react_sfmnorobot.csv"
# output_file = "exps/failed_cases_reward.csv"
# get_fail_cases(res_file, output_file)


res_file = "exps/results/evas/test_e007_rw100_1_1_react_sfmnorobot_noreward.csv"
output_file = "exps/failed_cases_noreward.csv"
get_fail_cases(res_file, output_file)