from itertools import count
from graph import *
from time import time
import random



class Ant:
    def __init__(self, 
        graph, 
        neighbors, 
        valid, 
        alpha=1, 
        beta=1, 
        q=0, 
        local_update=lambda _: None,
        termination=lambda self: bool(self.nodes)):
        """
        An ant is supposed to travel within a graph, visiting all nodes exactly once.
        :param graph: graph in which the ant is traveling
        :param neighbors: function that returns the neighbours of the node the ant currently is on
        :param valid: True once an ant has finished its solution
        :param alpha: importance of pheromone quantity
        :param beta: importance of heuristic information
        """
        self.graph = graph
        self.neighbors = lambda n: neighbors(self, n)
        self.valid = lambda: valid(self)
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.local_update = local_update
        self.termination = lambda: termination(self)
        self.reset()


    def __contains__(self, other):
        """
        Allows to test, whether a given ... is contained in either the nodes or edges within the current path.
        :param other: # TODO
        :return: True if ... already is in the path (either as a node or edge), False otherwise
        """
        if isinstance(other, Node):
            return other in self.path_nodes
        elif isinstance(other, Edge):
            return other in self.path_edges
        else:
            return False


    def reset(self):
        """
        Resets the current 'status' of an ant, i.e. ant is initialized on a random node within the graph and has not
        travelled yet.
        :return: ---
        """
        self.travel_dist = 0  # distance travelled
        self.node = random.choice(self.graph)  # current node on which the ant resides
        self.path_nodes = [self.node]  # current path of ant described through nodes
        self.path_edges = []  # current path of the ant described through edges


    def step(self):
        """
        Performs the next step of the ant, choosing the next node based on transition probabilities and adds the node
        and edges to the current path information.
        :return: True if a next step was performed, False otherwise
        """
        # Get the transition options and probabilities
        self.nodes, probabilities = self.transitions()
        # Stop if no nodes are to be visited anymore
        if not self.termination(): return False
        # Select a new node based on the provided probabilities and save the new node and the respective edge in the
        # path of the ant
        if random.random() > self.q:
            node = random.choices(self.nodes, weights=probabilities)[0]
        else:
            node = max(((n, p) for n, p in zip(self.nodes, probabilities)), key=lambda t: t[1])[0]
        edge = self.graph[self.node, node]
        self.path_nodes.append(node)
        self.path_edges.append(edge)
        self.travel_dist += edge.value  # Update the total travelled distance of this ant
        self.node = node  # Update the current node
        self.local_update(edge)
        return True


    def transitions(self):
        """
        Performs the calculations to receive information which nodes can be visited next with which probabilities.
        :return nodes: nodes that can be visited from the current node
        :return rho/ np.sum(rho): probabilities that the respective node is the next visited node
        """
        # For all nodes the current node is connected to, get the heuristic information and pheromone value for the
        # respective edge
        h = 1e-8
        nodes, heuristic, pheromone = [], [], []
        for node in self.node:
            if self.neighbors(node):
                edge = self.graph[self.node, node]
                nodes.append(node)
                heuristic.append(edge.heuristic)
                pheromone.append(edge.pheromone)
        # Perform the calculations to receive the probabilities to determine the next node
        eta = np.array(heuristic) + h
        tau = np.array(pheromone) + h
        psi = tau ** self.alpha * eta ** self.beta  # TODO this is not the same rho as in the Colony, right? Please rename this according to your preference
        return nodes, psi/np.sum(psi)



class Colony:
    def __init__(self, n_ants, graph, rho=0.1, alpha=1, beta=1, q=0,
                 neighbors=lambda a, n: len(a.graph) > len(a.path_nodes) and n not in a.path_nodes 
                                     or len(a.graph) == len(a.path_nodes) and n == a.path_edges[0].source,
                 valid=lambda a: len(a.path_nodes) == len(a.graph) + 1):
        """
        An ant colony consists of a given number of ants, a graph in which the ants travel and parameter information on
        the algorithm: rho, alpha, and beta.
        :param n_ants: Number of ants within the colony
        :param graph: Graph with a problem to solve
        :param rho: Evaporation rate
        :param alpha: importance of pheromone quantity
        :param beta: importance of heuristic information
        :param neighbors: a function how the neighbours of a node are defined
        :param valid: a function that determines if ants have found a valid solution
        """
        self.n_ants = n_ants
        self.graph = graph
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.graph.edges.heuristic = 1 / (self.graph.edges.value + 0.00001)  # TODO avoid potential division by zero
        self.ants = [Ant(graph=self.graph, neighbors=neighbors, valid=valid, alpha=alpha, 
                         beta=beta, q=q, local_update=self.local_update) for _ in range(n_ants)]
        self.node = random.choice(graph)
        self.graph.edges.pheromone = 1.0


    def __call__(self, N=None, visualization=False):
        """
        Executes the ant colony optimization algorithm.
        :param N: number of iterations
        :param visualization: whether intermediate results should be visualized
        :return: # TODO
        """
        start = time()
        iterator = count() if N is None else range(N)
        self.min_length = float('inf')
        self.min_path = None
        for _ in iterator:
            # Reset all ants at the beginning of each iteration
            for ant in self:
                ant.reset()
            # Let ants create their solution and global_update pheromones based on those
            self.construct(visualization=visualization)
            self.daemon()
            self.global_update()
        end = time()
        print(f"RUN TIME: {round(end - start, 1)} sec")
        return self.min_path, self.min_length, self.graph.edges.pheromone


    def __iter__(self):
        """
        Allows to iterate over the ants in the colony
        :return: ---
        """
        for ant in self.ants:
            yield ant


    def construct(self, visualization=False):
        """
        Constructs ant solutions as long as all ants can take a further step.
        :param visualization: whether intermediate results should be visualized
        :return: ---
        """
        while any([ant.step() for ant in self]):
            if visualization:
                visualization()
                continue
                try:
                    visualization()
                except:
                    pass


    def daemon(self):
        """
        (Optionally) Daemon actions: Updates the best path. Can be overritten
        :return: ---
        """
        try:
            min_path, min_length = min(((ant.path_nodes, ant.travel_dist)
                for ant in self if ant.valid()), key=lambda t: t[1])
            if self.min_length > min_length:
                self.min_length = min_length
                self.min_path = list(min_path)
        except:
            pass


    def local_update(self, edge):
        pass



class AntSystem(Colony):
    def __init__(self, n_ants, graph, rho=0.1, alpha=1, beta=1, **kwargs):
        super().__init__(n_ants=n_ants, graph=graph, 
                         rho=rho, alpha=alpha, beta=beta, **kwargs)


    def global_update(self, F=lambda x: 1/x):
        """
        Updates pheromone trails after all ants have finished their travels during one iteration.
        :param F: function to create the multiplicative inverse of a given number
        :return: ---
        """
        # Get all paths that all ants have taken
        ants = [ant for ant in self if ant.valid()]
        # Update the pheromones on all edges based on the previous pheromones weighted through the evaporation rate, and
        # the newly added pheromones
        #for edge in self.graph.edges:
        for edge in set(edge for ant in ants for edge in ant.path_edges):
            S = sum(F(ant.travel_dist) for ant in ants if edge in ant)
            edge.pheromone = (1 - self.rho) * edge.pheromone + S 



class AntColonySystem(Colony):
    def __init__(self, n_ants, graph, rho=0.1, alpha=1, beta=1,
                 q=0.3, tau=1.0, phi=0.1, **kwargs):
        super().__init__(n_ants=n_ants, graph=graph, 
                         rho=rho, alpha=alpha, beta=beta, **kwargs)
        self.graph.edges.pheromone = tau
        self.tau = tau
        self.phi = phi


    def global_update(self, F=lambda x: 1/x):
        """
        Updates pheromone trails after all ants have finished their travels during one iteration.
        :param F: function to create the multiplicative inverse of a given number
        :return: ---
        """
        try:
            ant = min((ant for ant in self if ant.valid()), key=lambda a: a.travel_dist)
            for edge in ant.path_edges:
                edge.pheromone = edge.pheromone * (1 - self.rho) + self.rho * F(ant.travel_dist)
        except:
            pass


    def local_update(self, edge):
        edge.pheromone = edge.pheromone * (1 - self.phi) + self.phi * self.tau


