import os
from sim.environment import Buffer
from sim.data_loader import DataLoader
import matplotlib.pyplot as plt

# This script helps visualize the trajectories of all the pedestrians in the dataset
# By visualizing and experimenting, you can set the check_regions and start_end_pos
# for test_case_generation.py

plt.rcParams.update({'font.size': 48})
plt.rcParams.update({'figure.figsize': [12.0, 10.0]})

i = 0 # which dataset (idx of the following lists) to visualize
datasets = ['eth', 'eth', 'ucy', 'ucy', 'ucy']
dataset_idxes = [0, 1, 0, 1, 2]

check_regions = [[[2, 2], [8, 8]],
                 [[-1, -6], [5, 0]],
                 [[4.5, 2], [10.5, 8]],
                 [[4.5, 3], [10.5, 9]],
                 [[4.5, 3], [10.5, 9]]]

start_end_pos = [[[[-8,5], [15,5]],
                  [[15,5], [-8,5]],
                  [[5,0], [5,12.5]],
                  [[5,12.5], [5,0]]],

                 [[[2,-10.5], [2,4.5]],
                  [[2,4.5], [2,-10.5]],
                  [[5,-3], [-3,-3]],
                  [[-3,-3], [5,-3]]],

                 [[[-0.5,5], [15.5,5]],
                  [[15.5,5], [-0.5,5]],
                  [[7.5,8], [7.5,2]],
                  [[7.5,2], [7.5,8]]],

                 [[[-0.5,6], [16,6]],
                  [[16,6], [-0.5,6]],
                  [[7.5,9], [7.5,2.5]],
                  [[7.5,2.5], [7.5,9]]],

                 [[[0,6], [16,6]],
                  [[16,6], [0,6]],
                  [[7.5,13], [7.5,0]],
                  [[7.5,0], [7.5,13]]]]

buffer = Buffer()
data = DataLoader(datasets[i], dataset_idxes[i], base_path="sim")
buffer = data.update_buffer(buffer)

all_x = []
all_y = []
for f in buffer.video_position_matrix:
    for p in f:
        all_x.append(p[0])
        all_y.append(p[1])

start_end = start_end_pos[i]
region = check_regions[i]

ax = plt.gca()
plt.scatter(all_x, all_y, s=1, marker='.')

pt1 = tuple(start_end[0][0])
pt2 = tuple(start_end[1][0])
pt3 = tuple(start_end[2][0])
pt4 = tuple(start_end[3][0])
ax.add_patch(plt.Circle(pt1, 0.5, color='r'))
ax.add_patch(plt.Circle(pt2, 0.5, color='r'))
ax.add_patch(plt.Circle(pt3, 0.5, color='r'))
ax.add_patch(plt.Circle(pt4, 0.5, color='r'))
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], lw=4, color='r')
plt.plot([pt3[0], pt4[0]], [pt3[1], pt4[1]], lw=4, color='r')

region_x = [region[0][0], region[0][0], region[1][0], region[1][0], region[0][0]]
region_y = [region[0][1], region[1][1], region[1][1], region[0][1], region[0][1]]
plt.plot(region_x, region_y, lw=4, color='k')

ax.set_aspect('equal', adjustable='box')
#ax.set_xlabel('X coordinates (m)')
#ax.set_ylabel('Y coordinates (m)')
plt.draw()
os.makedirs("data", exist_ok=True)
plt.savefig(os.path.join("data", datasets[i] + "_" + str(dataset_idxes[i]) + ".jpg"))
plt.show()
