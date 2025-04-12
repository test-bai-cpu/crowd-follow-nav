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