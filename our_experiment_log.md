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
        - run experiments, results: the add reward weight 1 worse than no reward, worse than add reward 5.

## Maj 4
 - Create traj_6.csv, same setup as traj_5.csv, but with vertical flow only.
 - Use both traj_5 and traj_6 to create synthetic_train7.json.
 - Use synthetic_train7.json, to run a version, with reward 1,5,0. Train version h5
   - The results shows the training curve does not look so good. But it is okay. we have good traning curve for other setups. 
  
## Maj 5
 - Use train version h5, to test on the eth/ucy data. It is quite okay. Have similar performance as others, even not train on eth/ucy data. Can combine with eth and ucy data to train together. Maybe performance can improve

Running: 
 - Try to only use the model train on the horizontal data, to test on the eth/ucy data. to see how bad it is, for curious. **exp_name**: e010_rw100_1_5_react_sfmrobot_c0.1_h4

TODO: 
 - Increase the safety threshold from 0.1 to 0.5, retrain the model. 

## Maj 6
 - As the results in "exps/results_ethucy/summary_results_eth_3.csv", the h4 version, which trained only on horital data, perform bad in eth ucy. But h5 version, which trained on both horizontal and vertical data, behave okay. Although the not use reward function performs better than reward 5 version.
 - So in the final train data, need to include both synthetic data and eth ucy data. threhold 0.5. 
 - Create a final train data
   - Create a all_v2.json, include test_case_generation_forRL.py, set sample 7 start and goal points, to make total line is around 41000, 2958 cases, which has similar length as synthetic_train7.json.
   - copy and paste the ones in synthetic_train7.json to all_v2.json. And use all_v2.json for train.
 - Running
   - Exp: run it, set reward to 0,1,5,(10). Set collision from 0.1 to 0.5. 
     - Potential issues: 
       - The diff is not quite obvious, may chance start and goal position in the synthetic dataset
       - Still have the problem of robot collide with new appear pedestrian? 
  
## Maj 7
 - Add distance in the RL reward group follow score computation.
 - Try with the distance version, in deepgearc57 and desktop, with weight 0,1,5. 
   - Cannot see much difference in the training process. Maybe should not combine two datasets in train and test. Maybe should separate, only eth/ucy in train/test, and only synthetic in train/test. And provide the train curve for both. But maybe current setting can have best performance.
   - Good! The success rate is good, and the demos looks more smooth and make sense, follow group to goal. Before it is a little jumpy? like exceed the group and go curve, not it is more like slow down to wait for go together with group. It has the join behavious by add the distance, which meet the expectation.
   - Bad! See some behaviours: cannot directly go to goal, but turn around a little and go to goal. If in the end the result for navigation time is not good enough, can look into this issue??? [TODO]

## Maj 12
 - Finished running using only the eth/ucy dataset. The result is weight 20 > weight 10 > weight 0. Stopped in all the machines.
 - Start to run using the synthetic data.
    - Create a synthetic_train8.json, which: compared to train7 version, change the start_end sample region, to move from side to middle, i.e. 7.5.
    - 



## Discussion Maj 6
 - In chaos case, FRP happens with baseline, 
 - Baseline: 
   - SARL
   - SARL + group reward
   - MPC
   - MPC + group cost
   - Follow is all you need paper
   - GO-MPC

 - Evaluation:
   - 

# Ideas wait to be tested
 - In the group-follow-score, should we add the distance to group also? Like to increase reward, need to decrease the distance between robot and crowd
 - Add verticals and horizontals flows in synthetic data, or even more angles. Use this data to train the model, to see if works on eth/ucy test data
 - If not works well, can train both with synthetic data and eth / ucy data.
 - If cannot see good results on our model / bad results on baseline model, use synthetic test data to create challenging cases for testing, like more density of crowds.