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



python3 main_paral.py --group --record --react --num_envs 4 --exp_name e001_10human_rlB256UTD1

python3 main_eval.py --group --record --react --animate --exp_name e001_10human_rlB256UTD1_fixSuccess --rl_model_weight n_samples_0300000

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