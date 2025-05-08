#!/bin/bash

# python3 main.py --group --record --react --animate
# python3 main.py --group --record --animate --react
# python3 main.py --group --record --react
python3 main.py --group --record --react --exp_name etest
python3 main.py --group --record --react --exp_name e001_10human_rlB512UTD1
python3 main.py --group --record --react --exp_name e001_10human_rlB256UTD1_fixSuccess
python3 main.py --group --record --react --exp_name e001_10human_rlB256UTD1RB20LR001
python3 main.py --group --record --react --exp_name e001_10human_rlB256UTD4_fixSuccess
python3 main.py --group --record --react --exp_name e001_10human_rlB256UTD4_mpccost  #(UTD1LD128)
python3 main.py --group --record --react --exp_name e001_10human_rlB256UTD1LD512_mpccost
python3 main.py --group --record --react --exp_name e001_10human_rlB256UTD1LD32_mpccost
python3 main.py --group --record --react --exp_name e002_10human_rlB256UTD1LD32_mpccost_FollowRangeFix  # For fixing the bug of rearrange the follow point from (-1,1) to (-3,3)
python3 main.py --group --record --react --exp_name e003_followPosOnly_samplempc  # For using sampling-based MPC and only output action: x and y
python3 main.py --group --record --react --exp_name e003_followPosOnly_samplempc_rot3  # max_rot_degrees = 3.14 and mpc_horizon = 10
python3 main.py --group --record --react --exp_name e003_followPosOnly_samplempc_rot3_rlRB100LR0003  # 0409 RL: RB100W,lr 0.0003
python3 main.py --group --record --react --exp_name e003_followPosOnly_samplempc_onlyGoal  # mpc cost to only follow and RL reward to only goal sparse and dense
python3 main.py --group --record --react --exp_name e003_followPosOnly_samplempc_eth0lefttoRight  # mpc cost to only follow and RL reward to only goal sparse and dense
python3 main.py --group --record --react --exp_name e004_l2r_goalvxvy  # mpc cost to only follow and RL reward to only goal sparse and dense
python3 main.py --group --record --react --exp_name e004_l2r_goalvxvy_mpcMultiStep  # try to run mpc 10 steps for one RL step
python3 main.py --group --record --react --exp_name e005_all  # last try multistep succeed easily, Now try with all test cases, all maps with all directions
python3 main.py --group --record --react --exp_name e005_all_mpcSafeCost  # last try multistep succeed easily, Now try with all test cases, all maps with all directions
python3 main.py --group --record --react --exp_name e005_all_mpcSafeCost_rlReward  # last try multistep succeed easily, Now try with all test cases, all maps with all directions
python3 main.py --group --record --react --exp_name e005_all_mpcSafeCost_rlReward_fw2  # set reward follow weight is 2
python3 main.py --group --record --react --exp_name e006_rw10_0.1_10  # set reward for sparse goal, dense goal, follow to 10, 0.1, 10
python3 main.py --group --record --exp_name e006_rw100_1_1_noreact  # set reward for sparse goal, dense goal, follow to 10, 0.1, 10
python3 main.py --group --record --exp_name e006_rw100_1_1_noreact_noreward  # set reward for sparse goal, dense goal, follow to 10, 0.1, 10



python3 main.py --group --record --react --exp_name e006_rw100_1_1_react
python3 main.py --group --record --react --exp_name e006_rw100_1_1_react_noreward
python3 main.py --group --record --react --exp_name e007_rw100_1_1_react_sfmnorobot # use social force: sfm_step_norobot
python3 main.py --group --record --react --exp_name e007_rw100_1_1_react_orcanorobot # use social force: orca_step_norobot

python3 main.py --group --record --react --exp_name e007_rw100_1_1_react_sfmnorobot_noreward # use social force: sfm_step_norobot
python3 main.py --group --record --react --exp_name e007_rw100_1_1_react_orcanorobot_noreward # use social force: orca_step_norobot


### Test synthetic version in e008
python3 main.py --group --record --react --exp_name e008_rw100_1_1_react_sfmrobot_reward --dset-file datasets_syn.yaml ### for using traj_1
python3 main.py --group --record --react --exp_name e008_rw100_1_1_react_sfmrobot_reward2 --dset-file datasets_syn.yaml ### for using traj_2
python3 main.py --group --record --react --exp_name e008_rw100_1_1_react_sfmrobot_reward2_w5 --dset-file datasets_syn.yaml ### for using traj_2, using weight 5 for group follow reward
python3 main.py --group --record --react --exp_name e008_rw100_1_1_react_sfmrobot_noreward --dset-file datasets_syn.yaml ### for using traj_1
python3 main.py --group --record --react --exp_name e008_rw100_1_1_react_sfmrobot_noreward2 --dset-file datasets_syn.yaml ### for using traj_2

### Found something wrong last time when generating synthetic test cases, the start_goal position is wrong. This time
### we correct it and add the vertical ones as well. So train on both horizontal and vertical ones.
### And change the collision threshold from 0.1 to 0.5
python3 main.py --group --record --react --exp_name e010_rw100_1_1_react_sfmrobot --dset-file datasets_syn.yaml # group reward is 1
python3 main.py --group --record --react --exp_name e010_rw100_1_1_react_sfmrobot_c0.5 --dset-file datasets_syn.yaml # group reward is 1
python3 main.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot --dset-file datasets_syn.yaml # group reward is 5
python3 main.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot_c0.5 --dset-file datasets_syn.yaml # group reward is 5, for change the collision threshold back to 0.5
python3 main.py --group --record --react --exp_name e010_rw100_1_0_react_sfmrobot --dset-file datasets_syn.yaml # group reward is 0, no group reward

python3 main.py --group --record --react --exp_name e010_rw100_1_1_react_sfmrobot_c0.5_h --dset-file datasets_syn.yaml # only train on horizontal flows
python3 main.py --group --record --react --exp_name e010_rw100_1_0_react_sfmrobot_c0.5_h --dset-file datasets_syn.yaml # only train on horizontal flows
python3 main.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot_c0.5_h --dset-file datasets_syn.yaml # only train on horizontal flows
python3 main.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot_c0.5_h_test --dset-file datasets_syn.yaml # only train on horizontal flows

python3 main.py --group --record --react --exp_name e010_rw100_1_1_react_sfmrobot_c0.5_h3 --dset-file datasets_syn.yaml --follow-weight 1 # use synthetic_train3.json
python3 main.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot_c0.5_h3 --dset-file datasets_syn.yaml --follow-weight 5 # use synthetic_train3.json
python3 main.py --group --record --react --exp_name e010_rw100_1_0_react_sfmrobot_c0.5_h3 --dset-file datasets_syn.yaml --follow-weight 0 # use synthetic_train3.json

python3 main.py --group --record --react --exp_name e010_rw100_1_1_react_sfmrobot_c0.1_h4 --dset-file datasets_syn.yaml --follow-weight 1 # use synthetic_train6.json
python3 main.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot_c0.1_h4 --dset-file datasets_syn.yaml --follow-weight 5 # use synthetic_train6.json
python3 main.py --group --record --react --exp_name e010_rw100_1_0_react_sfmrobot_c0.1_h4 --dset-file datasets_syn.yaml --follow-weight 0 # use synthetic_train6.json

python3 main.py --group --record --react --exp_name e010_rw100_1_1_react_sfmrobot_c0.1_h5 --dset-file datasets_syn.yaml --follow-weight 1 # use synthetic_train6.json
python3 main.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot_c0.1_h5 --dset-file datasets_syn.yaml --follow-weight 5 # use synthetic_train6.json
python3 main.py --group --record --react --exp_name e010_rw100_1_0_react_sfmrobot_c0.1_h5 --dset-file datasets_syn.yaml --follow-weight 0 # use synthetic_train6.json


### Start to use both synthetic and eth ucy data for training.
python3 main.py --group --record --react --exp_name e011_rw100_1_0_react_sfmrobot_c0.5_h6 --dset-file datasets_all.yaml --follow-weight 0 --collision_radius 0.5 # use all_v2.json
python3 main.py --group --record --react --exp_name e011_rw100_1_1_react_sfmrobot_c0.5_h6 --dset-file datasets_all.yaml --follow-weight 1 --collision_radius 0.5 # use all_v2.json
python3 main.py --group --record --react --exp_name e011_rw100_1_5_react_sfmrobot_c0.5_h6 --dset-file datasets_all.yaml --follow-weight 5 --collision_radius 0.5 # use all_v2.json

## Add the distance score
python3 main.py --group --record --react --exp_name e011_rw100_1_5_react_sfmrobot_c0.5_h6_distance --dset-file datasets_all.yaml --follow-weight 5 --collision_radius 0.5 # use all_v2.json

## same code, go back to only train on eth/ucy
python3 main.py --group --record --react --exp_name e011_rw100_1_5_react_sfmrobot_c0.5_h7_eth --dset-file datasets.yaml --follow-weight 5 --collision_radius 0.5 # use all_v2.json
python3 main.py --group --record --react --exp_name e011_rw100_1_0_react_sfmrobot_c0.5_h7_eth --dset-file datasets.yaml --follow-weight 0 --collision_radius 0.5 # use all_v2.json
python3 main.py --group --record --react --exp_name e011_rw100_1_1_react_sfmrobot_c0.5_h7_eth --dset-file datasets.yaml --follow-weight 1 --collision_radius 0.5 # use all_v2.json
python3 main.py --group --record --react --exp_name e011_rw100_1_10_react_sfmrobot_c0.5_h7_eth --dset-file datasets.yaml --follow-weight 10 --collision_radius 0.5 # use all_v2.json
python3 main.py --group --record --react --exp_name e011_rw100_1_20_react_sfmrobot_c0.5_h7_eth --dset-file datasets.yaml --follow-weight 20 --collision_radius 0.5 # use all_v2.json


python3 main_paral.py --group --record --react --num_envs 4 --exp_name e001_10human_rlB256UTD1

python3 main_eval.py --group --record --react --animate --exp_name e001_10human_rlB256UTD1_fixSuccess --rl_model_weight n_samples_0300000

python3 main_eval.py --group --record --react --animate --exp_name e001_10human_rlB256UTD1LD32_mpccost --rl_model_weight n_samples_0500000
python3 main_eval.py --group --record --react --animate --exp_name e002_10human_rlB256UTD1LD32_mpccost_FollowRangeFix --rl_model_weight n_samples_2000000

python3 main_eval.py --group --record --react --animate --exp_name e003_followPosOnly_samplempc --rl_model_weight n_samples_100000
python3 main_eval.py --group --record --react --animate --exp_name e003_followPosOnly_samplempc_rot3 --rl_model_weight n_samples_2200000
python3 main_eval.py --group --record --react --animate --exp_name e003_followPosOnly_samplempc_eth0lefttoRight --rl_model_weight n_samples_0600000
python3 main_eval.py --group --record --react --animate --exp_name e004_l2r_goalvxvy --rl_model_weight n_samples_0900000
python3 main_eval.py --group --record --react --animate --exp_name e004_l2r_goalvxvy_mpcMultiStep --rl_model_weight n_samples_0800000
python3 main_eval.py --group --record --react --animate --exp_name e005_all --rl_model_weight n_samples_0800000 ### For 
python3 main_eval.py --group --record --react --animate --exp_name e005_all_mpcSafeCost_rlReward --rl_model_weight n_samples_0300000 ### For 
python3 main_eval.py --group --record --react --animate --exp_name e005_all_mpcSafeCost_rlReward_fw5 --rl_model_weight n_samples_1200000 ### For 

#### For quantitative evaluation
python3 main_eval.py --group --record --react --exp_name e005_all_mpcSafeCost_rlReward_fw5 --rl_model_weight n_samples_1300000 ###  
python3 main_eval.py --group --record --react --exp_name e005_all_mpcSafeCost_rlReward --rl_model_weight n_samples_1200000 ### 
python3 main_eval.py --group --record --react --exp_name e005_all_mpcSafeCost --rl_model_weight n_samples_1000000 ### 


# python3 main_eval.py --group --record --react --animate --exp_name e007_rw100_1_1_react_sfmnorobot --rl_model_weight n_samples_0500000 --output-dir exps/results_time/e007_rw100_1_1_react_sfmnorobot
python3 main_eval.py --group --record --react --exp_name e007_rw100_1_1_react_sfmnorobot --rl_model_weight n_samples_0500000 --output-dir exps/results_collision_nonewappearped/e007_rw100_1_1_react_sfmnorobot
# python3 main_eval.py --group --record --react --exp_name e007_rw100_1_1_react_sfmnorobot_noreward --rl_model_weight n_samples_0500000 --output-dir exps/results_time/e007_rw100_1_1_react_sfmnorobot_noreward
python3 main_eval.py --group --record --react --animate --exp_name e007_rw100_1_1_react_sfmnorobot_noreward --rl_model_weight n_samples_0500000 --output-dir exps/results_collision_nonewappearped/e007_rw100_1_1_react_sfmnorobot_noreward


### Use previous model to test on synthetic dataset
python3 main_eval.py --group --record --react --animate --exp_name e008_rw100_1_1_react_sfmrobot_reward --rl_model_weight n_samples_0500000 --output-dir exps/results_synthetic/e008_rw100_1_1_react_sfmrobot_reward --dset-file datasets_syn.yaml
python3 main_eval.py --group --record --react --exp_name e008_rw100_1_1_react_sfmrobot_reward2 --rl_model_weight n_samples_0100000 --output-dir exps/results_synthetic/e008_rw100_1_1_react_sfmrobot_reward2 --dset-file datasets_syn.yaml
python3 main_eval.py --group --record --react --animate --exp_name e008_rw100_1_1_react_sfmrobot_reward2 --rl_model_weight n_samples_0500000 --output-dir exps/results_synthetic/e008_rw100_1_1_react_sfmrobot_reward --dset-file datasets_syn.yaml ### for using traj_2

## Test1 
python3 main_eval.py --group --record --react --animate --exp_name e010_rw100_1_1_react_sfmrobot_c0.1_h5 --rl_model_weight n_samples_0100000 --output-dir exps/results_synthetic/e010_rw100_1_1_react_sfmrobot_c0.1_h5 --dset-file datasets_syn.yaml --follow-weight 1
python3 main_eval.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot_c0.1_h5 --rl_model_weight n_samples_0100000 --output-dir exps/results_ethucy/e010_rw100_1_1_react_sfmrobot_c0.1_h5 --dset-file datasets.yaml --follow-weight 5
python3 main_eval.py --group --record --react --exp_name e010_rw100_1_5_react_sfmrobot_c0.1_h4 --rl_model_weight n_samples_0100000 --output-dir exps/results_ethucy/e010_rw100_1_5_react_sfmrobot_c0.1_h4 --dset-file datasets.yaml --follow-weight 5


### For both train on eth and synthetic:
python3 main_eval.py --group --record --react --animate --exp_name e011_rw100_1_5_react_sfmrobot_c0.5_h6 --rl_model_weight n_samples_0100000 --output-dir exps/results_all/e011_rw100_1_5_react_sfmrobot_c0.5_h6 --dset-file datasets_all.yaml --follow-weight 5 --collision_radius 0.5 # use all_v2.json
python3 main_eval.py --group --record --react --animate --exp_name e011_rw100_1_5_react_sfmrobot_c0.5_h6_distance --rl_model_weight n_samples_0100000 --output-dir exps/results_all/e011_rw100_1_5_react_sfmrobot_c0.5_h6_distance --dset-file datasets_all.yaml --follow-weight 5 --collision_radius 0.5 # use all_v2.json



### For plotting images
python3 main_eval.py --group --record --react --animate --exp_name e008_rw100_1_1_react_sfmrobot_reward2 --rl_model_weight n_samples_0100000 --output-dir exps/results_synthetic/check_imgs --dset-file datasets_syn.yaml

### Run baselines
# 1.for only use MPC and linear predictor
python3 main_eval_mpc.py --group --record --react --exp_name e009_mpc_linear --dset-file datasets_syn.yaml


results/e002_10human_rlB256UTD1LD32_mpccost_FollowRangeFix/n_samples_1500000

# for atc_file_num in {1..20..1}
# do
#     python3 check_scenarios.py --group --record --animate --dset-file datasets_atc.yaml --atc-file-num "$atc_file_num"
# done


#### For running program in desktop
cd /home/yufei/research/crowd-follow-nav
source .myvenv/bin/activate
python3 main.py --group --record --react --exp_name e000



#### For debugging
python3 main_debug.py --group --record --react --exp_name test1



#### For opening tensorboard locally
tensorboard --logdir results --port 6008