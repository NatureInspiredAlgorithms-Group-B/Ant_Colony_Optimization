from graph import Graph
from aco import AntColonyOpt as ACO
import random
import numpy as np
import matplotlib.pyplot as plt


def eval_aco(aco, graph):
    '''
    checks how long an ant needs from a specific start point to the goal to compare the optimization process over time.
    :param aco:
    :param graph:
    :return:
    '''
    dists = []
    nodes = [(1,1), (5,9), (3,5), (9,9), (8,1)]
    for i in nodes:
        dists.append(aco.test(start=i, goal=graph.goal))
    return dists


if __name__ == "__main__":

    # define algorithm parameters
    num_ants = 20
    evaporation_rate = 0.05
    initial_pheromone = 0.01
    num_iterations = 100
    alpha = 0.8
    beta = 0.6

    width = 10
    height = 10

    #characters = "abcdefghijklmnopqrstuvwxyz123456789"
    #grid = np.array([char for char in characters]).reshape((5, 7))

    # Initialize the test_dict with the connections within a grid
    test_dict = {}
    for i in range(0, width):
        for j in range(0, height):

            node = (i,j)
            test_dict[node] = []

            # 'connect' current node with the left, right, up, down neighbours
            if i >= 1:
                test_dict[node].append((i-1, j))
            if j >= 1:
                test_dict[node].append((i, j-1))
            if i < width-1:
                test_dict[node].append((i+1, j))
            if j < height-1:
                test_dict[node].append((i, j+1))

    goal = (3,3)
    test_graph = Graph(test_dict, goal=goal, pheromone_value=initial_pheromone)
    test_ACO = ACO(test_graph, evaporation_rate, alpha, beta)

    test_dists = []

    # start ant colony optimization
    for i in range(0, num_iterations):

        if i % 10 == 0:
            test_dists.append(eval_aco(test_ACO, test_graph))

        print("Iteration ", i)
        test_ACO.tmp = Graph(test_dict)
        #foodsources = ["c", "l", "z", "5"]

        # in each iteration, each ant is supposed to find the food source starting from a random position
        for _ in range(0, num_ants):
            start = random.choice(test_graph.get_nodes())
            test_ACO.create_path(start=start, foodsources=[], goal=goal)
        test_ACO.pheromone_update()

        # img = np.zeros(grid.shape).astype(np.float64)
        # for (i, j) in test_ACO.graph.get_pheromones().keys():
        #     img[grid == j] += test_ACO.graph.get_pheromones()[(i, j)]
        #
        # plt.imshow(img, cmap='gray')
        # plt.show()

    final_pheromones = test_ACO.graph.get_pheromones()

    # visualize the process of the optimization process thorugh the length of the path for the same start and end points
    test_dists.append(eval_aco(test_ACO, test_graph))
    test_dists = np.asarray(test_dists)
    x_pts = range(len(test_dists))
    marker = ['r.', 'b.', 'g.', 'y.', 'm.']
    test_dists = np.swapaxes(test_dists, 0, 1)
    for i in range(5):
        plt.plot(x_pts, test_dists[i], marker[i])
    plt.grid()
    plt.show()

    #img = np.zeros(grid.shape).astype(np.float64)
    #for (i, j) in final_pheromones.keys():
     #   img[grid == j] += final_pheromones[(i, j)]

    #plt.imshow(img, cmap='gray')
    #plt.show()




