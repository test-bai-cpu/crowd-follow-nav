## Apr 9
 - Run expriment, for only let RL guiding to goal. not consider anything with humans. Remove human part both from MPC cost and also RL reward.
 - 


## Apr 10
 - Check the vx = vy = 0 problem, make sure mpc_horizon=10 and change the MPC cost function, to set follow cost as the minimum distance of the path to the goal. Before it is the last pos of MPC to goal. Goal mean follow point.
 - Set up the deepgear, and put crowd-nav docker on it, can run the code now. Running the from left to right version. Which only train on data, from eth0, start is left, goal is right. 
 - 

## Apr 11
 - run, which only heading to goal, does not consider humans, and only in eth0 scene, only from left to right, experiments, cannot have good results
 - Also we run the experiments, for change dimension from 42 to 44, for adding the goal vx, vy, which is -robotvx, -robotvy

## Apr 23
 - check the results from previous day, find that follow point bouncing too much, decide to run a version which run mpc 10 times, then generate one RL follow point. e004_l2r_goalvxvy_mpcMultiStep 