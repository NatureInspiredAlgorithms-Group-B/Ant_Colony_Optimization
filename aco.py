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
        """
        probability of which node of the graph to choose next
        :return: next node
        """

        if curr_node in self.graph.nodes: # TODO what exactly is this trying to test?
            # for every possible next node: calculate the probability to move there from curr_node
            tmp_probs = {}
            final_probs = {}

            # Get the
            for i in self.graph.graph[curr_node]:
                tmp_probs[i] = (self.graph.get_pheromone(curr_node, i) ** self.alpha * self.heurisitic_info(curr_node, i, path) ** self.beta)
            sum_probs = sum(tmp_probs.values())

            # Get the probabilities to select the next node
            for i in self.graph.graph[curr_node]:
                final_probs[i] = tmp_probs[i] / sum_probs

            # Return the node with the maximal probability, if there are multiple with the same maximum probability,
            # return a random one of those
            max_nodes = [node for node, prob in zip(final_probs.keys(), final_probs.values()) if prob == max(final_probs.values())]

            return random.choice(max_nodes)

            # if random.random() < 0.99:
            #     return random.choice(max_nodes)
            # else:
            #     return random.choice(list(final_probs.keys()))

    def heurisitic_info(self, i, j, path):
        """
        function representing prior knowledge about the specific knowledge about the edge from node i to j
        :return:
        """
        # If traverse from i to j has been part of the best yet solution, it is probably a good traverse
        # factor = 1
        # if is_contained(path, [i,j]):
        #     factor = 4

        # Punish going in circles
        if j in path: # Punish going in circles TODO this is stupid if we want to run in circles :D
            return 0.001

        return 1

    def pheromone_update(self):
        """
        updating all pheromones depending on evaporation rate and gained experience
        """
        for key in self.graph.graph.keys():
            for value in self.graph.graph.get(key):
                self.graph.pheromones[(key, value)] = (1-self.e_rate) * self.graph.get_pheromone(key, value) + self.get_new_pheromones(key, value)

    def get_new_pheromones(self, i, j):
        """
        returns the new amount of pheromones of this part of the solution
        :return:
        """
        # If node was visisted, return the update
        if (i, j) in self.tmp.pheromones.keys():
            return self.tmp.pheromones[(i, j)]
        else:
            return 0 # No update for this node

    def eval_solution(self, path, targets):
        """
        evaluates one solution path depending on the problem and defines the amount of pheromones for each part of the path
        influences the function get_new_pheromones for the pheromone update
        :return:
        """

        if path:
            start = np.argwhere(self.graph.grid == path[0])[0]
            dist = 0

            # Add up the distance travelled between the foodsources
            for target in targets:
                end = np.argwhere(self.graph.grid == target)[0] # First time meeting that node is counted
                dist += get_euclidean_distance(start, end)

                # reset start to to current node
                start = end.copy()

            # From last foodsource to goal
            end = np.argwhere(self.graph.grid == self.graph.goal)[-1]
            dist += get_euclidean_distance(start, end)

            # Update pheromones -> penalize long paths with respect to the euclidean distance
            for i in range(len(path) - 1):
                self.tmp.pheromones[(path[i], path[i + 1])] += round(0.05 * (dist / len(path)), 4)

    def create_path(self, start, foodsources, goal):
        """
        creates path in the graph from start node to goal node depending on the transition function
        :param foodsources:
        :param start: node
        :param goal: node
        :return: sequence of nodes
        """
        curr_node = start
        path = [curr_node]
        visited_foodsource = []

        # Visit all foodsources first
        while foodsources:
            curr_node = self.transition_probability_fct(curr_node, path)  # path for heuristics
            path.append(curr_node)
            if curr_node in foodsources:
                visited_foodsource.append(curr_node)
                foodsources.remove(curr_node)

        # Then resturn to the nest
        while not curr_node == goal:
            curr_node = self.transition_probability_fct(curr_node, path) # path for heuristics
            path.append(curr_node)

        # Update the currently best solution
        if self.best_solution:
            if len(path) < len(self.best_solution):
                self.best_solution = path
        else:
            self.best_solution = path # If not initialized before

        self.eval_solution(path, visited_foodsource)


