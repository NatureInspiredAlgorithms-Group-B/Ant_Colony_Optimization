import numpy as np



class View:
    DEFAULT_ATTRIBUTES = ['type', 'reference']
    def __init__(self, reference):
        self.reference = reference
        self.type = type(reference)
        assert any(isinstance(self.reference, structure) for structure in (dict, list, set, tuple))


    def __getitem__(self, key):
        return self.reference[key]


    def __setitem__(self, key, val):
        self.reference[key] = val


    def __getattr__(self, attr):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if isinstance(self.reference, dict):
                return self.reference[attr]    
            else:
                return self.type(getattr(item, attr) for item in self.reference)
        else:
            return super().__getattr__(attr)


    def __setattr__(self, attr, val):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if isinstance(self.reference, dict):
                self.reference[attr] = val  
            else:
                for item in self.reference:
                    setattr(item, attr, val)
        else:
            super().__setattr__(attr, val)


    def __str__(self):
        return str(self.reference)


    def __repr__(self):
        return 'View ' + str(self)



class Node:
    DEFAULT_ATTRIBUTES = ['graph', 'reference']
    def __init__(self, graph, reference):
        self.graph = graph
        self.reference = reference


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
            return f"Node ({', '.join(key + ':' + str(getattr(self, key)) for key in self.graph._node_values.keys())})"
        else:
            return f"[{self.reference}]"


    def __str__(self):
        if self.graph._node_values:
            return f"[{self.reference}:({', '.join(str(getattr(self, key)) for key in self.graph._node_values.keys())})]"
        else:
            return f"[{self.reference}]"


    def __int__(self):
        return self.reference


    def __eq__(self, other):
        return self.reference == other.reference and self.graph == other.graph



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
            except:
                self.graph._edge_values[attr][int(self.source)][int(self.target)]
        else:
            super().__setattr__(attr, val)


    def __repr__(self):
        return f"Edge {str(self)} ({', '.join(key + ':' + str(getattr(self, key)) for key in self.graph._edge_values.keys())})"


    def __str__(self):
        return f"<{self.source}⟼{self.target}>"


    def __eq__(self, other):
        return self.source == other.source and self.target == other.target and self.graph == other.graph



class Graph:
    def __init__(self, nodes=None, edges=None, default_value=0):
        self.default_value = default_value
        self._node_values = {}
        self._edge_values = {}
        # CONSTRUCTION
        if nodes is not None:
            if isinstance(nodes, int):
                self._nodes = [Node(self, n) for n in range(nodes)]
                self._edges = np.full((n_nodes, n_nodes), default_value)
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
            return View(self._nodes)
        elif attr == 'edges':
            return View(self._edge_values)
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


    def _registery(self, mode, attr, val=None):
        val = val if val is not None else self.default_value
        numeral = lambda x: any(isinstance(x, c) for c in (int, float, complex, bool))
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
    def __init__(self, n_nodes, min_distance=0, max_distance=1):
        distances = np.random.rand(n_nodes, n_nodes)
        distances = min_distance + distances * (max_distance - min_distance)
        distances[np.eye(n_nodes).astype(bool)] = 0
        super().__init__(edges=distances)


    def route(self):
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
        
"""


if __name__ == '__main__':
    #G = Graph(edges=np.ones((4, 4)))
    #N = G[0]
    #E = G[0, 1]
    #print(N, E)
    #print(G.nodes.reference)
    #G.nodes.l = []
    #print(G.nodes.l)

    G = TSP(4)
    print(G)
    print(G.route())
