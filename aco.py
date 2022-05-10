from itertools import count
import random


class Ant:
    def __init__(self, graph, neighbors, valid, alpha=1, beta=1):
        """

        :param graph: graph in which the ant is traveling
        :param neighbors: the nodes the ant can travel to next
        :param valid:
        :param alpha:
        :param beta:
        """
        self.graph = graph
        self.neighbors = lambda n: neighbors(self, n)
        self.valid = lambda: valid(self)
        self.alpha = alpha
        self.beta = beta
        self.reset()

    def __contains__(self, other):
        """
        Tests, whether a given ... is contained in either the nodes or edges within the current path.
        :param other:
        :return:
        """
        if isinstance(other, Node):
            return other in self.path_nodes
        elif isinstance(other, Edge):
            return other in self.path_edges
        else:
            return False

    def reset(self):
        """
        Resets the current 'status' of an ant, i.e. ant is initialized in a random node within the graph and has not
        travelled yet.
        :return:
        """
        self.length = 0 # distance travelled
        self.node = random.choice(self.graph) # current node on which the ant resides
        self.path_nodes = [self.node] # current path of ant described through nodes
        self.path_edges = [] # current path of the ant described through edges

    def step(self):
        """
        Performs the next step of the ant, choosing the next node based on transition probabilities and adds the node
        and edges to the current path information.
        :return:
        """
        # Get the transition options and probabilities
        nodes, probabilities = self.transitions()

        # Stop if no nodes are to be visited anymore
        if not nodes:
            return False

        # Select a new node based on the provided probabilities and save the new node and the respective edge in the
        # path of the ant
        node = random.choices(nodes, weights=probabilities)[0]
        edge = self.graph[self.node, node]
        self.path_nodes.append(node)
        self.path_edges.append(edge)
        self.length += edge.value # Update the travelled distance
        self.node = node # Update the current node

        return True

    def transitions(self):
        """
        Performs the calculations to receive information which nodes can be visited next with which probabilities.
        :return nodes: nodes that can be visited from the current node
        :return rho/ np.sum(rho): probabilities that the respective node is the next visited node
        """
        # For all nodes the current node is connected to, get the heuristic information and pheromone value for the
        # respective edge
        nodes, heuristic, pheromone = [], [], []
        for node in self.node:
            if self.neighbors(node):
                edge = self.graph[self.node, node]
                nodes.append(node)
                heuristic.append(edge.heuristic)
                pheromone.append(edge.pheromone)

        # Perform the calculations to receive the probabilities to determine the next node
        eta = np.array(heuristic)
        tau = np.array(pheromone)
        rho = tau ** self.alpha * eta ** self.beta
        return nodes, rho/np.sum(rho)


class AntColony:
    def __init__(self, n_ants, graph, rho=0.1, alpha=1, beta=1,
                 neighbors=lambda a, n: len(a.graph) > len(a.path_nodes) and n not in a.path_nodes 
                                     or len(a.graph) == len(a.path_nodes) and n == a.path_edges[0].source,
                 valid=lambda a: len(a.path_nodes) == len(a.graph) + 1):
        """
        An ant colony consists of a given number of ants, a graph in which the ants travel and ...
        :param n_ants:
        :param graph:
        :param rho:
        :param alpha:
        :param beta:
        :param neighbors:
        :param valid:
        """
        self.n_ants = n_ants
        self.graph = graph
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.graph.edges.heuristic = 1/self.graph.edges.value
        self.ants = [Ant(self.graph, neighbors, valid, alpha, beta) for _ in range(n_ants)]
        self.node = random.choice(graph)
        self.graph.edges.pheromone = 1.0

    def __call__(self, N=None, visualization=False):
        """
        Executes the ant colony optimization algorithm.
        :param N:
        :param visualization:
        :return:
        """
        iterator = count() if N is None else range(N)
        self.min_length = float('inf')
        self.min_path = None
        for _ in iterator:
            for ant in self.ants:
                ant.reset()
            self.construct(visualization=visualization)
            self.daemon()
            self.update()
        return self.min_path, self.min_length

    def __iter__(self):
        """

        :return:
        """
        for ant in self.ants:
            yield ant


    def construct(self, visualization=False):
        """
        Constructs ant solutions
        :param visualization:
        :return:
        """
        while any(ant.step() for ant in self):
            if visualization:
                self.visualize()


    def daemon(self):
        """(Optionally) Daemon Actions"""
        min_path, min_length = min(((ant.path_nodes, ant.length)
            for ant in self if ant.valid()), key=lambda t: t[1])
        min_ant = min(self, key=lambda t: t.length)
        if self.min_length > min_length:
            self.min_length = min_length
            self.min_path = list(min_path)


    def update(self, F=lambda x: 1/x):
        """
        Updates pheromone trails
        :param F:
        :return:
        """
        ants = [ant for ant in self if ant.valid()]
        paths = [ant.path_edges for ant in self if ant.valid()]
        for edge in set(edge for ant in ants for edge in ant.path_edges):
            S = sum(F(ant.length) for ant in ants if edge in ant)
            edge.pheromone = (1 - self.rho) * edge.pheromone + self.rho * S


if __name__ == '__main__':
    from graph import *

    G = TSP(6)
    C = AntColony(10, G, alpha=1, beta=10, rho=0.1)
    search_path, search_length = G.route()
    ant_path, ant_length = C(100)
    print(G.edges.value)
    print(ant_path, search_path)
    print(ant_length, search_length)


