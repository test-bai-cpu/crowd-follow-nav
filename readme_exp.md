# setups 


1. max_follow_pos_delta: to define the follow pos can be how far away.

```python
    max_follow_pos_delta = (mpc_config.getint('mpc_env', 'mpc_horizon') *
                            mpc_config.getfloat('mpc_env', 'max_speed'))
```
This valus is used for normalization of the follow pos. Because follow pos is a relative value to current robot pos.

2. 