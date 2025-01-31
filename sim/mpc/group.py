import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

SOCIAL_SHAPE_DENSITY = 20

def _DBScan_grouping(labels, properties, standard):
    # DBSCAN in-group clustering among grouped labels
    # Inputs:
    # labels: the input labels. This will be destructively updated to 
    #         reflect the group memberships after DBSCAN.
    # properties: the input that clustering is based on.
    #             Could be positions, speed or orientation.
    # standard: the threshold value for clustering.
    # Outputs:
    # labels: the updated group membership after DBSCAN. 
    #         The group membership is encoded as a positive integer.

    max_lb = max(labels)
    for lb in range(max_lb + 1):
        sub_properties = []
        sub_idxes = []
        # Only perform DBSCAN within groups (i.e. have the same membership id)
        for i in range(len(labels)):
            if labels[i] == lb:
                sub_properties.append(properties[i])
                sub_idxes.append(i)
    
        # If there's only 1 person then no need to further group
        if len(sub_idxes) > 1:
            # Note min_samples set to 1 so every point will be assigned a group
            # e.g. no points will be treated as noise
            db = DBSCAN(eps = standard, min_samples = 1)
            sub_labels = db.fit_predict(sub_properties)
            max_label = max(labels)

            # db.fit_predict always return labels starting from index 0
            # we can add these to the current biggest id number to create 
            # new group ids.
            for i, sub_lb in enumerate(sub_labels):
                if sub_lb > 0:
                    labels[sub_idxes[i]] = max_label + sub_lb
    return labels

def grouping(position_array, velocity_array, params):
    # Grouping based on position, velocity and orientation
    # Inputs:
    # position_array: the positions of all people
    # velocity_array: the velocities of all people
    # params: the parameters for grouping
    # Outputs:
    # labels: the group membership of each person

    num_people = len(position_array)
    vel_orientation_array = []
    vel_magnitude_array = []
    for [vx, vy] in velocity_array:
        velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)
        if velocity_magnitude < params['spd_ignore']:
            # if too slow, then treated as being stationary
            vel_orientation_array.append((0.0, 0.0))
            vel_magnitude_array.append((0.0, 0.0))
        else:
            vel_orientation_array.append((vx / velocity_magnitude, vy / velocity_magnitude))
            vel_magnitude_array.append((0.0, velocity_magnitude)) # Add 0 to fool DBSCAN
    # grouping in current frame (three passes, each on different criteria)
    labels = [0] * num_people
    labels = _DBScan_grouping(labels, vel_orientation_array,
                                params['th_threshold'])
    labels = _DBScan_grouping(labels, vel_magnitude_array,
                                params['spd_threshold'])
    labels = _DBScan_grouping(labels, position_array,
                                params['pos_threshold'])
    return labels

def boundary_dist(velocity, rel_ang, const=0.354163, offset=0):
    # Parameters are from Rachel Kirby's thesis
    # This function calculates the boundary distance of a pedestrian
    # given the velocity and the relative angle.
    
    front_coeff = 1.0
    side_coeff = 2.0 / 3.0
    rear_coeff = 0.5
    safety_dist = 0.5
    velocity_x = velocity[0]
    velocity_y = velocity[1]

    velocity_magnitude = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
    variance_front = max(0.5, front_coeff * velocity_magnitude)
    variance_side = side_coeff * variance_front
    variance_rear = rear_coeff * variance_front

    rel_ang = rel_ang % (2 * np.pi)
    flag = int(np.floor(rel_ang / (np.pi / 2)))
    if flag == 0:
        prev_variance = variance_front
        next_variance = variance_side
    elif flag == 1:
        prev_variance = variance_rear
        next_variance = variance_side
    elif flag == 2:
        prev_variance = variance_rear
        next_variance = variance_side
    else:
        prev_variance = variance_front
        next_variance = variance_side

    dist = np.sqrt(const / ((np.cos(rel_ang) ** 2 / (2 * prev_variance)) + (np.sin(rel_ang) ** 2 / (2 * next_variance))))
    dist = max(safety_dist, dist)
    
    dist = max(dist - offset, 1e-9) # Avoid negative distance

    return dist

def draw_social_shapes(position, velocity, const, offset=0):
    # This function draws social group shapes
    # given the positions and velocities of the pedestrians in a group.
    #
    # Inputs:
    # position: the positions of the pedestrians
    # velocity: the velocities of the pedestrians
    # const: the constant for calculating the boundary distance
    # offset: the offset to be reducted from the boundary distance
    #
    # Outputs:
    # convex_hull_vertices: the vertices of the convex hull of the social group

    total_increments = SOCIAL_SHAPE_DENSITY # controls the resolution of the blobs
    angle_increment = 2 * np.pi / total_increments

    # Draw a personal space for each pedestrian within the group
    contour_points = []
    for i in range(len(position)):
        center_x = position[i][0]
        center_y = position[i][1]
        velocity_x = velocity[i][0]
        velocity_y = velocity[i][1]
        velocity_angle = np.arctan2(velocity_y, velocity_x)

        # Draw four quater-ovals with the axis determined by front, side and rear "variances"
        # The overall shape contour does not have discontinuities.
        for j in range(total_increments):

            rel_ang = angle_increment * j
            value = boundary_dist(velocity[i], rel_ang, const, offset)
            addition_angle = velocity_angle + rel_ang
            x = center_x + np.cos(addition_angle) * value
            y = center_y + np.sin(addition_angle) * value
            contour_points.append([x, y])

    # Get the convex hull of all the personal spaces
    convex_hull_vertices = []
    hull = ConvexHull(np.array(contour_points))
    for i in hull.vertices:
        hull_vertice = (contour_points[i][0], contour_points[i][1])
        convex_hull_vertices.append(hull_vertice)

    return convex_hull_vertices

def draw_all_social_spaces(gp_labels, positions, velocities, const, offset=0):
    # This function draws social group shapes for all groups
    # given the positions, velocities and group labels of the pedestrians.
    #
    # Inputs:
    # gp_labels: the group labels of the pedestrians
    # positions: the positions of the pedestrians
    # velocities: the velocities of the pedestrians
    # const: the constant for calculating the boundary distance
    # offset: the offset to be reducted from the boundary distance
    #
    # Outputs:
    # all_vertices: the vertices of the convex hull of the social groups

    all_vertices = []
    all_labels = np.unique(gp_labels)
    for curr_label in all_labels:
        group_positions = []
        group_velocities = []
        for i, l in enumerate(gp_labels):
            if l == curr_label:
                group_positions.append(positions[i])
                group_velocities.append(velocities[i])
        vertices = draw_social_shapes(group_positions, group_velocities, const, offset)
        all_vertices.append(vertices)
    return all_vertices