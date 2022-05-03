from graph import Graph
from aco import AntColonyOpt as ACO
import random
import numpy as np

if __name__ == "__main__":

    # define algorithm parameters
    num_ants = 20
    evaporation_rate = 0.2
    initial_pheromone = 0
    num_iterations = 100
    alpha = 0.6
    beta = 0.8

    grid = np.array([["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]])

    # initialize the graph depending on the problem
    test_dict = {"a": ["b", "d"],
                 "b": ["a", "e", "c"],
                 "c": ["b", "f"],
                 "d": ["a", "e", "g"],
                 "e": ["b", "d", "f", "h"],
                 "f": ["c", "e", "i"],
                 "g": ["d", "h"],
                 "h": ["g", "e", "i"],
                 "i": ["h", "f"]}
    #test_phero = {("a","b"): 0.01, ("b","a"): 0.01, ("b","c"): 0.01, ("b","d"): 0.01, ("c","d"): 0.01}
    goal = "d"

    test_graph = Graph(test_dict, goal=goal, pheromone_value=0.01, grid=grid)
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



