import numpy as np
from graph import Graph
import math
import random


class AntColonyOpt:

    def __init__(self, graph=Graph(), e_rate=0.5, alpha=0.5, beta=0.5):
        self.graph = graph
        self.e_rate = e_rate
        self.alpha = alpha
        self.beta = beta
        self.tmp = {}

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
                tmp_probs[i] = (self.graph.get_pheromone(curr_node, i) ** self.alpha * self.heurisitc_info(curr_node, i, path) ** self.beta)
            sum_probs = sum(tmp_probs.values())

            # Get the probabilities to select the next node
            for i in self.graph.graph[curr_node]:
                final_probs[i] = tmp_probs[i] / sum_probs

            # Return the node with the maximal probability, if there are multiple with the same maximum probability,
            # return a random one of those
            max_nodes = [node for node, prob in zip(final_probs.keys(), final_probs.values()) if prob == max(final_probs.values())]

            return random.choice(max_nodes)

    def heurisitc_info(self, i, j, path):
        '''
        function representing prior knowledge about the specific knowledge about the edge from node i to j
        :return:
        '''

        if j in path: # Punish going in circles
            return 0.001
        return 1

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
            # Distance between ant start (path[0]) and goal (euclidean distance in grid)
            start = np.argwhere(self.graph.grid == path[0])[0]
            end = np.argwhere(self.graph.grid == self.graph.goal)[0]

            dist_x = start[0] - end[0]
            dist_y = start[1] - end[1]

            dist = math.sqrt(dist_x ** 2 + dist_y ** 2)

            # Update pheromones -> penalize longer paths
            for i in range(len(path) - 1):
                self.tmp.pheromones[(path[i], path[i + 1])] += round((1. / dist) * .02, 4)
                #self.tmp.pheromones[(path[i+1], path[i])] += round((1. / dist) * .02, 4) # TODO update in both directions yes/ no?


    def create_path(self, start, goal):
        '''
        creates path in the graph from start node to goal node depending on the transition function
        :param start: node
        :param goal: node
        :return: sequence of nodes
        '''
        curr_node = start
        path = [curr_node] # TODO We start with the start node in the path, right?
        while not curr_node == goal:
            curr_node = self.transition_probability_fct(curr_node, path) # path for heuristics
            path.append(curr_node)
        print(path)
        self.eval_solution(path)


