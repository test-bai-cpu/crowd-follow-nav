# setups 


1. max_follow_pos_delta: to define the follow pos can be how far away.

```python
    max_follow_pos_delta = (mpc_config.getint('mpc_env', 'mpc_horizon') *
                            mpc_config.getfloat('mpc_env', 'max_speed'))
```
This valus is used for normalization of the follow pos. Because follow pos is a relative value to current robot pos.




# DOCKER

```bash
docker pull dockerhuaniden/crowd-nav-dev:latest

git pull https://github.com/test-bai-cpu/crowd-follow-nav.git

# Datasets: ewap_dataset, ucy_dataset put in crowd-follow-nav/sim

docker run -it --rm \
  -v /home/yufei/research/crowd-follow-nav:/crowd-follow-nav \
  crowd-nav-dev
```


# Run in deepgear now

docker run -it --rm \
  --gpus all \
  --name crowd-follower \
  -v /home/yzu/crowd-follow-docker/crowd-follow-nav:/workdir/crowd-follow-nav \
  dockerhuaniden/crowd-nav-dev

cd crowd-follow-nav

python3 main.py --group --record --react --exp_name e004_l2r_goalvxvy_mpcMultiStep

ctrl p q

