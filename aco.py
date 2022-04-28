import numpy as np
from graph import Graph


class AntColonyOpt:

    def __init__(self, graph=Graph(), e_rate=0.5, alpha=0.5, beta=0.5):
        self.graph = graph
        self.e_rate = e_rate
        self.alpha = alpha
        self.beta = beta

    def transition_probability_fct(self, curr_node):
        '''
        probability of which node of the graph to choose next
        :return: next node
        '''

        if curr_node in self.graph:
            # for every possible next node: calculate the probability to move there from curr_node
            tmp_probs = {}
            final_probs = {}
            for i in self.graph.get(curr_node):
                tmp_probs[i] = (self.graph.get_pheromone(curr_node, i) ** self.alpha * self.heurisitc_info(curr_node, i) ** self.beta)
            sum_probs = sum(tmp_probs.values())
            for i in self.graph.get(curr_node):
                final_probs[i] = tmp_probs[i] / sum_probs
            return max(final_probs, key=final_probs.get)

    def heurisitc_info(self, i, j):
        '''
        function representing prior knowledge about the specific knowledge about the edge from node i to j
        :return:
        '''
        pass

    def pheromone_update(self):
        '''
        updating all pheromones depending on evaporation rate and gained experience
        '''
        for key in self.graph.keys():
            for value in self.graph.get(key):
                self.graph.pheromones[(key, value)] = (1-self.e_rate) * self.graph.get_pheromone(key, value) + self.get_new_pheromones(key, value)

    def get_new_pheromones(self, i, j):
        '''
        returns the new amount of pheromones of this part of the solution
        :return:
        '''
        pass

    def eval_solution(self, path):
        '''
        evaluates one solution path depending on the problem and defines the amount of pheromones for each part of the path
        influences the function get_new_pheromones for the pheromone update
        :return:
        '''
        pass

    def create_path(self, start, goal):
        '''
        creates path in the graph from start node to goal node depending on the transition function
        :param start: node
        :param goal: node
        :return: sequence of nodes
        '''
        curr_node = start
        path = []
        while not curr_node == goal:
            curr_node = self.transition_probability_fct(curr_node)
            path.append(curr_node)
        self.eval_solution(path)


