from graph import Graph
from aco import AntColonyOpt as ACO
import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # define algorithm parameters
    num_ants = 20
    evaporation_rate = 0.8
    initial_pheromone = 0.1
    num_iterations = 1000
    alpha = 0.6
    beta = 0.8

    # characters = "abcdefghijklmnopqrstuvwxyz123456789"
    #
    # grid = np.array([char for char in characters]).reshape((5, 7))
    #
    # # Initialize the test_dict with the connections within a grid
    # test_dict = {}
    # for i in range(grid.shape[0]):
    #     for j in range(grid.shape[1]):
    #
    #         node = grid[i, j]
    #         test_dict[node] = []
    #
    #         # 'connect' current node with the left, right, up, down neighbours
    #         if i >= 1:
    #             test_dict[node].append(grid[i-1, j])
    #         if j >= 1:
    #             test_dict[node].append(grid[i, j-1])
    #         if i < grid.shape[0]-1:
    #             test_dict[node].append(grid[i+1, j])
    #         if j < grid.shape[1]-1:
    #             test_dict[node].append(grid[i, j+1])
    #
    #
    #
    # start = "3"
    # goal = "g"

    cities = ['a', 'b', 'c', 'd', 'e']

    test_dict = {}
    for city1 in cities:
        test_dict[city1] = []

        for city2 in cities:
            if not city1 == city2:
                test_dict[city1].append(city2)

    print(test_dict)

    distances = {}
    for start_city in test_dict.keys():

        for end_city in test_dict[start_city]:

            if not (start_city, end_city) in distances.keys() and not (end_city, start_city) in distances.keys():
                distance = random.randrange(10, 200)
                distances[start_city, end_city] = distance
                distances[end_city, start_city] = distance

    distances = {('a', 'b'): 5,
                 ('b', 'a'): 5,
                 ('b', 'c'): 10,
                 ('c', 'b'): 10,
                 ('c', 'd'): 15,
                 ('d', 'c'): 15,
                 ('d', 'a'): 400,
                 ('a', 'd'): 400,
                 ('a', 'c'): 5,
                 ('c', 'a'): 5,
                 ('d', 'b'): 20,
                 ('b', 'd'): 20,
                 ('a', 'e'): 30,
                 ('e', 'a'): 30,
                 ('b', 'e'): 300,
                 ('e', 'b'): 300,
                 ('c', 'e'): 25,
                 ('e', 'c'): 25,
                 ('d', 'e'): 80,
                 ('e', 'd'): 80}

    print(distances)

    test_graph = Graph(test_dict, distance_dict=distances, pheromone_value=initial_pheromone)
    test_ACO = ACO(test_graph, evaporation_rate, alpha, beta)

    # start ant colony optimization
    for i in range(0, num_iterations):

        print("Iteration ", i)
        test_ACO.tmp = Graph(test_dict)

        # in each iteration, each ant is supposed to find the food source starting from a random position
        for _ in range(0, num_ants):
            path = test_ACO.create_path(list(test_dict.keys()))
            print(path)
        test_ACO.pheromone_update()

    final_pheromones = test_ACO.graph.get_pheromones()

    print(final_pheromones)




