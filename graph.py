class Graph:

    def __init__(self, graph={}, pheromones={}, pheromone_value=0, goal="", grid=[]):
        '''
        Weighted, directed Graph representation of the problem.
        :param graph:
        :param pheromones:
        '''
        self.graph = graph
        self.goal = goal
        self.nodes = list(self.graph.keys())
        if pheromones:
            self.pheromones = pheromones
        else:
            self.pheromones = {}
            # initialize the pheromones of all edges to a small positive value
            for key in self.graph.keys():
                for value in self.graph.get(key):
                    self.pheromones[(key, value)] = pheromone_value
        self.grid = grid

    def get_pheromone(self, i, j):
        '''
        return amount of pheromones from node i to node j
        :param i: from node i
        :param j: to node j
        :return: pheromone of this edge
        '''
        return self.pheromones[(i,j)]

    def get_nodes(self):
        return self.nodes

    def get_pheromones(self):
        return self.pheromones
