import numpy as np
from graph import Graph
import math
import random

from utils import get_euclidean_distance, is_contained


class AntColonyOpt:

    def __init__(self, graph=Graph(), e_rate=0.5, alpha=0.5, beta=0.5):
        self.graph = graph
        self.e_rate = e_rate
        self.alpha = alpha
        self.beta = beta
        self.tmp = {}
        self.best_solution = []

    def transition_probability_fct(self, curr_node, path):
        '''
        probability of which node of the graph to choose next
        :return: next node
        '''

        if curr_node in self.graph.nodes: # TODO what exactly is this trying to test?
            # for every possible next node: calculate the probability to move there from curr_node
            tmp_probs = {}
            final_probs = {}

            # Get the
            for i in self.graph.graph[curr_node]:

                if i in path:
                    tmp_probs[i] = 0 # exclude nodes that have been visited
                else:
                    tmp_probs[i] = (self.graph.get_pheromone(curr_node, i) ** self.alpha * self.heuristic_info(curr_node, i, path) ** self.beta)
            sum_probs = sum(tmp_probs.values())

            if sum_probs == 0.0:
                sum_probs = 0.000001 # avoid division by zero

            # Get the probabilities to select the next node
            for i in self.graph.graph[curr_node]:
                final_probs[i] = tmp_probs[i] / sum_probs

            return random.choices(list(final_probs.keys()), weights=list(final_probs.values()))[0]

    def heuristic_info(self, i, j, path):
        '''
        function representing prior knowledge about the specific knowledge about the edge from node i to j
        :return:
        '''

        # Punish going in circles
        punish = 0.0001 if j in path else 1
        return punish / self.graph.distance_dict[(i, j)]

    def pheromone_update(self):
        '''
        updating all pheromones depending on evaporation rate and gained experience
        '''
        for key in self.graph.graph.keys():
            for value in self.graph.graph.get(key):
                self.graph.pheromones[(key, value)] = (1-self.e_rate) * self.graph.get_pheromone(key, value) + self.get_new_pheromones(key, value)

    def get_new_pheromones(self, i, j):
        '''
        returns the new amount of pheromones of this part of the solution
        :return:
        '''
        # If node was visisted, return the update
        if (i, j) in self.tmp.pheromones.keys():
            return self.tmp.pheromones[(i, j)]
        else:
            return 0 # No update for this node

    def eval_solution(self, path):
        '''
        evaluates one solution path depending on the problem and defines the amount of pheromones for each part of the path
        influences the function get_new_pheromones for the pheromone update
        :return:
        '''

        if path:

            dist = 0
            for i in range(len(path)-1):
                dist += self.graph.distance_dict[(path[i], path[i+1])]

            print(dist)

            # Update pheromones -> penalize long paths with respect to the euclidean distance
            for i in range(len(path) - 1):
                self.tmp.pheromones[(path[i], path[i + 1])] += round((4 / dist), 4)

    def create_path(self, to_be_visited):
        """

        :param to_be_visited:
        :return:
        """
        # Start with a random node
        curr_node = random.choice(to_be_visited)
        path = [curr_node]

        # Then return to the nest
        while True:

            # Delete the node from those nodes that still have to be visited, if there are none, the path is completed
            if curr_node in to_be_visited:
                to_be_visited.remove(curr_node)
                if not to_be_visited:
                    break

            curr_node = self.transition_probability_fct(curr_node, path)
            path.append(curr_node)

        # Return to start node
        path.append(path[0])

        self.eval_solution(path)

        return path


