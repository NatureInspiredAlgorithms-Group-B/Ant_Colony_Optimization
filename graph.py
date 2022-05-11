from copy import deepcopy
from math import prod
import numpy as np
import matplotlib.pyplot as plt



numeral = lambda x: any(isinstance(x, c) for c in (int, float, complex, bool))



class View:
    DEFAULT_ATTRIBUTES = ['type', 'reference', 'mode', 'graph']
    THRESHOLD = 1e-4

    def __init__(self, mode, graph):
        self.mode = mode
        self.graph = graph
        self.type = type(self.reference)
        assert any(isinstance(self.reference, structure) for structure in (dict, list, set, tuple))


    def __getitem__(self, key):
        return self.reference[key]


    def __setitem__(self, key, val):
        self.reference[key] = val


    def __getattr__(self, attr):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if isinstance(self.reference, dict):
                if attr not in self.reference:
                    self.graph._registery(mode=self.mode, attr=attr)
                return self.reference[attr]
            else:
                if attr not in self.reference:
                    self.graph._registery(mode=self.mode, attr=attr)
                return self.type(getattr(item, attr) for item in self.reference)
        elif attr == 'reference':
            if self.mode == 'node':
                return self.graph._node_values
            elif self.mode == 'edge':
                return self.graph._edge_values
        else:
            return super().__getattr__(attr)


    def __setattr__(self, attr, val):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if isinstance(self.reference, dict):
                if numeral(val):
                    self.reference[attr] = np.full_like(self.value, val).astype(type(val))
                elif isinstance(val, np.ndarray):
                    if np.sum(np.abs(val - val.T))/prod(val.shape) > self.THRESHOLD:
                        raise Exception("The TSP graph is undirected, but the np.matrix is not transposition invariant!")
                    self.reference[attr] = val
                elif self.mode == 'node' and (isinstance(val, tuple) or isinstance(val, list)) and len(val) == len(self.graph):
                    self.reference[attr] = val
                else:
                    if self.mode == 'node':
                        self.reference[attr] = [deepcopy(val) for _ in self.graph]
                    elif self.mode == 'edge':
                        self.reference[attr] = [[deepcopy(val) for _ in self.graph] for _ in self.graph]
            else:
                for item in self.reference:
                    setattr(item, attr, deepcopy(val))
        else:
            if numeral(val):
                super().__setattr__(attr, np.full_like(self.value, val).astype(type(val)))
            else:
                super().__setattr__(attr, val)


    def __str__(self):
        if self.mode == 'node':
            return 'Nodes[' + ', '.join(str(node) for node in self.graph) + ']'
        elif self.mode == 'edge':
            return 'Edges[' + ', '.join(str(self.graph[a, b]) for a in self.graph for b in self.graph if self.graph[a, b]) + ']'


    def __repr__(self):
        return 'View ' + str(self)



class Node:
    DEFAULT_ATTRIBUTES = ['graph', 'reference', 'name']

    def __init__(self, graph, reference, name=None):
        self.graph = graph
        self.reference = reference
        self.name = name or reference


    def __iter__(self):
        for node in self.graph:
            if self.graph[self, node]:
                yield node


    def __getattr__(self, attr):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if attr not in self.graph._node_values:
                self.graph._registery(mode='node', attr=attr)
            return self.graph._node_values[attr][self.reference]
        else:
            super().__getattr__(attr)
    

    def __setattr__(self, attr, val):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if attr not in self.graph._node_values:
                self.graph._registery(mode='node', attr=attr, val=val)
            self.graph._node_values[attr][self.reference] = val
        else:
            super().__setattr__(attr, val)


    def __repr__(self):
        if self.graph._node_values:
            return f"Node⟨{', '.join(key + ':' + str(getattr(self, key)) for key in self.graph._node_values.keys() if not key.startswith('_'))}⟩"
        else:
            return f"⟨{self.name}⟩"


    def __str__(self):
        return f"⟨{self.name}⟩"


    def __int__(self):
        return self.reference


    def __eq__(self, other):
        return self.reference == other.reference and self.graph == other.graph


    def __hash__(self):
        return hash(str(self))



class Edge:
    DEFAULT_ATTRIBUTES = ['graph', 'source', 'target']

    def __init__(self, graph, source, target):
        self.graph = graph
        self.source = source
        self.target = target


    def __getattr__(self, attr):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if attr not in self.graph._edge_values:
                self.graph._registery(mode='edge', attr=attr)
            try:
                return self.graph._edge_values[attr][int(self.source), int(self.target)]
            except:
                return self.graph._edge_values[attr][int(self.source)][int(self.target)]
        else:
            super().__getattr__(attr)


    def __setattr__(self, attr, val):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if attr not in self.graph._edge_values:
                self.graph._registery(mode='edge', attr=attr, val=val)
            try:
                self.graph._edge_values[attr][int(self.source), int(self.target)] = val
                if not self.graph.bidirectional:
                    self.graph._edge_values[attr][int(self.target), int(self.source)] = val
            except:
                self.graph._edge_values[attr][int(self.source)][int(self.target)] = val
                if not self.graph.bidirectional:
                    self.graph._edge_values[attr][int(self.target)][int(self.source)] = val
        else:
            super().__setattr__(attr, val)


    def __repr__(self):
        return f"Edge{str(self)} ⟨{', '.join(key + ':' + str(getattr(self, key)) for key in self.graph._edge_values.keys() if not key.startswith('_'))}⟩"


    def __str__(self):
        return f"{str(self.source)}⟝{str(self.target)}"


    def __eq__(self, other):
        directed = self.source == other.source and self.target == other.target and self.graph == other.graph
        undirected = self.source == other.target and self.target == other.source
        return directed or not self.graph.bidirectional and undirected


    def __hash__(self):
        return hash(str(self))



class Graph:
    def __init__(self, nodes=None, edges=None, default_value=0, bidirectional=False):
        """
        creates an instance of a Graph
        :param nodes: int, np.ndarray
        :param edges: None, np.ndarray
        :param default_value:
        """
        self.default_value = default_value
        self.bidirectional = bidirectional
        self._node_values = {}
        self._edge_values = {}
        # For visualization:
        self.bg_img = plt.imread("osm_germany.png")
        # CONSTRUCTION
        if nodes is not None:
            if isinstance(nodes, int):
                self._nodes = [Node(self, n) for n in range(nodes)]
                self._edges = np.full((len(nodes), len(nodes)), default_value)
            elif isinstance(nodes, np.ndarray):
                if isinstance(edges, np.ndarray):
                    if nodes.shape + nodes.shape == edges.shape:
                        self._nodes = nodes
                        self._edges = edges
                    else:
                        raise Warning(f"For nodes of shape (n,) edges must be of shape (n, n). Got {nodes.shape} and {edges.shape} instead!")
                else:
                    self._nodes = nodes
                    self._edges = np.full(nodes.shape + nodes.shape, default_value)
        elif isinstance(edges, np.ndarray):
            self._nodes = [Node(self, n) for n in range(edges.shape[0])]
            self._edges = edges
        else:
            self._nodes = []
            self._edges = None
        self.edges.value = edges


    def __getitem__(self, key):
        """
        returns Node or Edge or False corresponding to the key
        :param key: Node, int, Edge, (int, int)
        """
        if isinstance(key, int):
            try:
                return self._nodes[key]
            except:
                raise IndexError(f"Index {key} is out of range {len(self)}!")
        elif isinstance(key, tuple):
            if len(key) == 2:
                try:
                    source, target = key
                    if isinstance(source, int) and isinstance(target, int):
                        if self._edges[source, target]:
                            return Edge(self, self[source], self[target])
                        else:
                            return None
                    elif isinstance(source, Node) and isinstance(target, Node):
                        if self._edges[int(source), int(target)]:
                            return Edge(self, source, target)
                        else:
                            return None
                    else:
                        raise TypeError(f"Keys must be of type int or Node not {type(source), type(target)}")
                except:
                    raise IndexError(f"Index {key} is out of range {len(self), len(self)}!")
            else:
                raise IndexError(f"Index for Edges expects 2 keys but {len(key)} where given!")
        elif isinstance(key, Node) or isinstance(key, Edge):
            return key
        else:
            raise Exception(f"Index must be of type int, tuple, Node or Edge not {type(key)}")


    def __getattr__(self, attr):
        if attr == 'nodes':
            return View('node', self)
        elif attr == 'edges':
            return View('edge', self)
        else:
            raise NotImplemented(f"{attr} is currently not implemented")


    def __iter__(self):
        for node in self._nodes:
            yield node


    def __len__(self):
        return len(self._nodes)


    def __str__(self):
        return f"""GRAPH ({len(self)})
{self._edges}"""


    def __repr__(self):
        return str(self)


    def __contains__(self, other):
        if isinstance(other, Node):
            return other in self._nodes
        elif isinstance(other, Edge):
            return bool(self[other])
        else:
            return False


    def _registery(self, mode, attr, val=None):
        """
        All Graph data are stored in the Graph instance. 
        Attribute initialization on Nodes and Edges call 
        the registery to update the Graph data.
        :param mode: str ['node'|'edge']
        :param attr: str
        :param val:
        """
        val = val if val is not None else self.default_value
        if mode == 'node':
            if numeral(val):
                self._node_values[attr] = np.full((len(self),), type(val)())
            else:
                self._node_values[attr] = [type(val)() for _ in range(len(self))]
        elif mode == 'edge':
            if numeral(val):
                self._edge_values[attr] = np.full((len(self), len(self)), type(val)())
            else:
                self._edge_values[attr] = [[type(val)() for _ in range(len(self))] for _ in range(len(self))]



class TSP(Graph):
    def __init__(self, n_nodes=None, coordinates=None, min_distance=0, max_distance=1):
        if n_nodes is not None:
            distances = np.random.rand(n_nodes, n_nodes)
            distances = min_distance + distances * (max_distance - min_distance)
            distances = (distances + distances.T)/2
            distances[np.eye(n_nodes).astype(bool)] = 0
            super().__init__(edges=distances, bidirectional=False)
        elif coordinates is not None:
            distances = np.zeros((len(coordinates), len(coordinates)))
            coordinates = [np.array(c) for c in coordinates]
            for i, c_i in enumerate(coordinates):
                for j, c_j in enumerate(coordinates):
                    distances[i, j] = np.linalg.norm(c_i - c_j)
            super().__init__(edges=distances, bidirectional=False)
            self.nodes.coordinates = coordinates
        else:
            raise Exception("Either coordinates or n_nodes must be specified!")


    def route(self):
        """
        returns the shortest path as well as its length.
        """
        def rec(self, path, length):
            source = path[-1]
            min_length = float('inf')
            min_path = None
            for target in source:
                if target not in path:
                    edge = self[source, target]
                    rec_path, rec_length = rec(self, path + [target], length + edge.value)
                    if rec_length < min_length:
                        min_length, min_path = rec_length, rec_path
            if min_path is not None:
                return min_path, min_length
            else:
                return path + [path[0]], length + self[path[-1], path[0]].value
        return rec(self, [self[0]], 0)



class Germany(TSP):
    def __init__(self):
        cities = {'Osnabrück': (235, 234),
                  'Hamburg': (324, 137),
                  'Hanover': (312, 226),
                  'Frankfurt': (264, 391),
                  'Munich': (396, 528),
                  'Berlin': (478, 215),
                  'Leipzig': (432, 302),
                  'Düsseldorf': (175, 310),
                  'Kassel': (302, 305), 
                  'Cottbus': (521, 274), 
                  'Düsseldorf': (177, 311), 
                  'Bremen': (270, 173), 
                  'Karlsruhe': (251, 469),
                  'Nürnberg': (373, 437),
                  'Saarbrücken': (187, 452)}
        # coordinates for osna, hamburg, hanover, frankfurt, munich, berlin and leipzig, kassel, Düsseldorf
        coordinates = list(cities.values())
        super().__init__(coordinates=coordinates)


    def visualize(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('Route')
        if 'pheromone' in self._edge_values.keys():
            pheromone = self.edges.pheromone
        else:
            raise Warning(f"Edges of the graph must have the attribute 'pheromone'. Graph only got {', '.join(self._edge_values.keys())}")
        # Mark all cities with a dot
        coords = np.array([node.coordinates for node in self])
        ax.scatter(coords[:, 0], coords[:, 1])
        # Normalize the pheromone levels
        max_phero = np.max(pheromone[pheromone != 1.0])
        min_phero = np.min(pheromone[pheromone != 1.0])
        for n_i in self:
            for n_j in self:
                # Draw the connection of the two nodes based on the pheromone concentration
                if not n_i == n_j:
                    x_values = [n_i.coordinates[0], n_j.coordinates[0]]
                    y_values = [n_i.coordinates[1], n_j.coordinates[1]]
                    # Normalize pheromones for better visibility
                    normed_pheromone = (pheromone[n_i.name, n_j.name] - min_phero) / (max_phero - min_phero)
                    # Draw the connection based on determined pheromone level
                    ax.plot(x_values, y_values, '-', color=[0, 0, 0, normed_pheromone])
        ax.imshow(plt.imread("osm_germany.png"))
        plt.axis('off')
        plt.tight_layout()
        plt.show()



"""
USECASES:
    Graph[x] -> Node(x)
        x: index, Node
    Graph[x] -> Edge(x)
        x: Edge
    Graph[x, y] -> Edge(x, y)
        x: index, Edge

    Node(x).attr = val -> Graph[x].attr = val
        attr: known, unknown
        val: Any
    Edge(x, y).attr = val -> Graph[x, y].attr = val
        attr: known, unknown
        val: Any

    Node(x).attr -> Graph[x].attr
        âttr: known, unknown
    Edge(x, y).attr -> Graph[x, y].attr
        âttr: known, unknown

    Graph.nodes.attr -> [N0.attr, N1.attr ... Nn.attr]
        attr: known
    Graph.nodes.attr = [numeral, ..., numeral] -> Graph.nodes.attr = [N0.attr, N1.attr ... Nn.attr]
        attr: known
        
"""


if __name__ == '__main__':
    #G = Graph(edges=np.ones((4, 4)))
    #N = G[0]
    #E = G[0, 1]
    #print(N, E)
    #print(G.nodes.l)
    #G.nodes.l = []
    #print(G.nodes.l)
    #G[0].l.append(0)
    #print(G.nodes.l)
    G = TSP(4)
    edge = G[2, 3]
    m = np.ones((4, 4))
    m[:,0] = 1.00001
    G.edges.new_val = m
    print(edge)
    edge.pheromone = 42
    edge._hi = 0
    print(repr(edge))
    print(str(edge))
    print(G.edges)
    print(G.nodes)
    exit()
    print(G)
    print(G.route())
    G.edges.N = 10
    G.edges.N *= 4
    G[0, 1].N += 2
    print(G[0, 1].N, G[1, 2].N)
    G.nodes.N = 10
    G.nodes.N *= 4
    G[0].N += 2
    print(G[0].N, G[1].N)
