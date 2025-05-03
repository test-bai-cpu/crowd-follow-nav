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
 - check the results from previous day, find that follow point bouncing too much, decide to run a version which run mpc 10 times, then generate one RL follow point. e004_l2r_goalvxvy_mpcMultiStep. In this version, we also randomly select one case every time, and while true.


## Maj 2
 - Use synthetic data.
 - Test 1: run synthetic_train3.json. To test: whether start point / goal point for robot should be same as flow of human, of should be in the middle of two flows. 
    - Previous results are: 
        - Use one flow: middle, the other flow: same as flow position. Have good results. i.e., add weight5 group follow better than add weight1 better than add 0
        - Use two flow, middle. Have not good results, weight 1 is better than weight 5, better than weight 0. i.e. adding more group follow weight can lead to bad performance. But this may also make sense. 

## Maj 3
 - The h3 version have desired train curve. But not sure if the evaluation results meets the hypothesis or not. 
 - Think about reduce the flow density, and put the start and goal point in the middle? - v4
    - Generate traj_4.csv. Reduce group number from (5,11) to (2,6)
    - traj_5.csv: frames_per_group_base = 50 (100->50), n_groups = 100 (200 -> 100)
    - synthetic_train6.json, use traj_5.csv, but move start goal point from each flow to middle 7.5



# Ideas wait to be tested
 - In the group-follow-score, should we add the distance to group also? Like to increase reward, need to decrease the distance between robot and crowd
 - Add verticals and horizontals flows in synthetic data, or even more angles. Use this data to train the model, to see if works on eth/ucy test data
 - If not works well, can train both with synthetic data and eth / ucy data.
 - If cannot see good results on our model / bad results on baseline model, use synthetic test data to create challenging cases for testing, like more density of crowds.