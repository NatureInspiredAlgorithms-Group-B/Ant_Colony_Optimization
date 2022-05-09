from graph import Graph
from itertools import count
import numpy as np
import random



class Ant:
    def __init__(self, graph, neighbors, valid, alpha=1, beta=1, node=None):
        self.graph = graph
        self.neighbors = lambda n: neighbors(self, n)
        self.valid = lambda: valid(self, )
        self.node = node if node is not None else random.choice(graph)
        self.alpha = alpha
        self.beta = beta
        self.reset()


    def __contains__(self, other):
        if isinstance(other, Node):
            return other in self.path_nodes
        elif isinstance(other, Edge):
            return other in self.path_edges
        else:
            return False


    def reset(self):
        self.path_nodes = []
        self.path_edges = []
        self.length = 0


    def step(self):
        nodes, probabilities = self.transitions()
        if not nodes: return False
        node = random.choices(nodes, weights=probabilities)[0]
        edge = self.graph[self.node, node]
        self.path_nodes.append(node)
        self.path_edges.append(edge)
        self.length += edge.value
        self.node = node
        return True



    def transitions(self):
        nodes, heuristic, pheromone = [], [], []
        for node in self.node:
            if self.neighbors(node):
                edge = self.graph[self.node, node]
                nodes.append(node)
                heuristic.append(edge.heuristic)
                pheromone.append(edge.pheromone)
        eta = np.array(heuristic)
        tau = np.array(pheromone)
        rho = tau**self.alpha * eta**self.beta
        return nodes, rho/np.sum(rho)




class AntColony:
    def __init__(self, n_ants, graph, rho=0.1, alpha=1, beta=1,
                 neighbors=lambda a, n: n not in a.path_nodes,
                 valid=lambda a: len(a.path_nodes) == len(a.graph)):
        self.n_ants = n_ants
        self.graph = graph
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.graph.edges.heuristic = 1/self.graph.edges.value
        self.ants = [Ant(graph, neighbors, valid, alpha, beta) for _ in range(n_ants)]
        self.node = random.choice(graph)
        self.graph.edges.pheromone = 1.0


    def __call__(self, N=None, visualization=False):
        iterator = count() if N is None else range(N)
        self.min_length = float('inf')
        self.min_path = None
        for n in iterator:
            for ant in self.ants:
                ant.reset()
            self.construct(visualization=visualization)
            self.daemon()
            self.update()
        return self.min_path, self.min_length


    def __iter__(self):
        for ant in self.ants:
            yield ant


    def construct(self, visualization=False):
        """Constructs Ant Solutions"""
        while any(ant.step() for ant in self):
            if visualization:
                self.visualize()


    def daemon(self):
        """(Optionally) Daemon Actions"""
        min_path, min_length = min(((ant.path_nodes, ant.length)
            for ant in self if ant.valid()), key=lambda t: t[1])
        if self.min_length > min_length:
            print(min_length, min_path)
            self.min_length = min_length
            self.min_path = min_path



    def update(self, F=lambda x: 1/x):
        """Updates Pheromon Trails"""
        ants = [ant for ant in self if ant.valid()]
        paths = [ant.path_edges for ant in self if ant.valid()]
        for edge in set(edge for ant in ants for edge in ant.path_edges):
            S = sum(F(ant.length) for ant in ants if edge in ant)
            edge.pheromone = (1 - self.rho) * edge.pheromone + self.rho * S



if __name__ == '__main__':
    from graph import *

    G = TSP(6)
    C = AntColony(10, G, alpha=1, beta=10, rho=0.2)
    search_path, search_length = G.route()
    ant_path, ant_length = C(100)
    print(G.edges.value)
    print(ant_path, search_path)
    print(ant_length, search_length)

    summ = lambda s: sum(G[a, b].value for a, b in zip(s[:-1], s[1:]))

    print(summ(ant_path + [ant_path[0]]))
    print(summ(search_path))
    #print(G.edges)

