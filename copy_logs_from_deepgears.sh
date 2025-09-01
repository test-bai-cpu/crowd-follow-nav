#!/bin/bash


# scp -r yzu@130.243.124.183:~/crowd-follow-docker/crowd-follow-nav/results/logs/e011_rw100_1_0_react_sfmrobot_c0.5_h8_syn ~/research/crowd-follow-nav/results/logs
# scp -r yzu@130.243.124.183:~/crowd-follow-docker/crowd-follow-nav/results/logs/e011_rw100_1_20_react_sfmrobot_c0.5_h8_syn ~/research/crowd-follow-nav/results/logs


# scp -r yzu@130.243.124.57:~/crowd-follow-docker/crowd-follow-nav/results/logs/e011_rw100_1_1_react_sfmrobot_c0.5_h8_syn ~/research/crowd-follow-nav/results/logs
# scp -r yzu@130.243.124.57:~/crowd-follow-docker/crowd-follow-nav/results/logs/e011_rw100_1_5_react_sfmrobot_c0.5_h8_syn ~/research/crowd-follow-nav/results/logs


### copy logs from deepgear to desktop
# scp -r yzu@130.243.124.74:~/crowd-follow-docker/crowd-follow-nav/results/logs/e013_rw100_1_0_react_sfmrobot_c0.5_h9_eth ~/research/crowd-follow-nav/results/logs
# scp -r yzu@130.243.124.74:~/crowd-follow-docker/crowd-follow-nav/results/logs/e013_rw100_1_1_react_sfmrobot_c0.5_h9_eth ~/research/crowd-follow-nav/results/logs
# scp -r yzu@130.243.124.74:~/crowd-follow-docker/crowd-follow-nav/results/logs/e013_rw100_1_5_react_sfmrobot_c0.5_h9_eth ~/research/crowd-follow-nav/results/logs

# scp -r yzu@130.243.124.74:~/crowd-follow-docker/crowd-follow-nav/results/logs/e014* ~/research/crowd-follow-nav/results/logs

### copy models from deepgear to desktop

# mkdir -p ~/research/crowd-follow-nav/results/e013_rw100_1_0_react_sfmrobot_c0.5_h9_eth

# scp "yzu@130.243.124.74:~/crowd-follow-docker/crowd-follow-nav/results/e013_rw100_1_0_react_sfmrobot_c0.5_h9_eth/n_samples_*" ~/research/crowd-follow-nav/results/e013_rw100_1_0_react_sfmrobot_c0.5_h9_eth/


# mkdir -p ~/research/crowd-follow-nav/results/e013_rw100_1_1_react_sfmrobot_c0.5_h9_eth

# scp "yzu@130.243.124.74:~/crowd-follow-docker/crowd-follow-nav/results/e013_rw100_1_1_react_sfmrobot_c0.5_h9_eth/n_samples_*" ~/research/crowd-follow-nav/results/e013_rw100_1_1_react_sfmrobot_c0.5_h9_eth/


# mkdir -p ~/research/crowd-follow-nav/results/e013_rw100_1_5_react_sfmrobot_c0.5_h9_eth

# scp "yzu@130.243.124.74:~/crowd-follow-docker/crowd-follow-nav/results/e013_rw100_1_5_react_sfmrobot_c0.5_h9_eth/n_samples_*" ~/research/crowd-follow-nav/results/e013_rw100_1_5_react_sfmrobot_c0.5_h9_eth/


scp -r "yzu@130.243.124.74:~/crowd-follow-docker/crowd-follow-nav/results/e016*" ~/research/crowd-follow-nav/results/
scp -r "yzu@130.243.124.183:~/crowd-follow-docker/crowd-follow-nav/results/e016*" ~/research/crowd-follow-nav/results/
