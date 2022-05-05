import math
import numpy as np

def get_euclidean_distance(a, b):
    dist_x = a[0] - b[0]
    dist_y = a[1] - b[1]

    return math.sqrt(dist_x ** 2 + dist_y ** 2)

def is_contained(path, subpath):

    print(path)

    moves_1 = np.resize(np.array(path), (len(path) // 2, 2))
    moves_2 = np.resize(np.array(path[1:]), (len(path[1:]) // 2, 2))

    print(moves_1)

    all_moves = np.append(moves_1, moves_2).reshape(-1, 2)

    potential_subset = np.array(subpath)

    return np.any(np.all(all_moves == potential_subset, axis=1))



