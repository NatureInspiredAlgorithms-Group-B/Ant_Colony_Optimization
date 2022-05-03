from graph import Graph
from aco import AntColonyOpt as ACO
import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # define algorithm parameters
    num_ants = 20
    evaporation_rate = 0.2
    initial_pheromone = 0.01
    num_iterations = 200
    alpha = 0.6
    beta = 0.8

    characters = "abcdefghijklmnopqrstuvwx"

    grid = np.array([char for char in characters]).reshape((4, 6))

    # Initialize the test_dict with the connections within a grid
    test_dict = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):

            node = grid[i, j]
            test_dict[node] = []

            # 'connect' current node with the left, right, up, down neighbours
            if i >= 1:
                test_dict[node].append(grid[i-1, j])
            if j >= 1:
                test_dict[node].append(grid[i, j-1])
            if i < grid.shape[0]-1:
                test_dict[node].append(grid[i+1, j])
            if j < grid.shape[1]-1:
                test_dict[node].append(grid[i, j+1])

    goal = "d"
    test_graph = Graph(test_dict, goal=goal, pheromone_value=initial_pheromone, grid=grid)
    test_ACO = ACO(test_graph, evaporation_rate, alpha, beta)

    # start ant colony optimization
    for i in range(0, num_iterations):

        print("Iteration ", i)
        test_ACO.tmp = Graph(test_dict)

        # in each iteration, each ant is supposed to find the food source starting from a random position
        for _ in range(0, num_ants):
            start = random.choice(test_graph.get_nodes())
            test_ACO.create_path(start, goal)
        test_ACO.pheromone_update()
    final_pheromones = test_ACO.graph.get_pheromones()

    img = np.zeros(grid.shape).astype(np.float64)
    for (i, j) in final_pheromones.keys():
        img[grid == i] += final_pheromones[(i, j)]
        img[grid == j] += final_pheromones[(i,j)]

    plt.imshow(img, cmap='gray')
    plt.show()



