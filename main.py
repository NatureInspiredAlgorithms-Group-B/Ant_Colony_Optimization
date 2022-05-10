from graph import *
from aco import AntColony


if __name__ == '__main__':
    G = TSP(8)
    C = AntColony(10, G, alpha=1, beta=1, rho=0.1)
    search_path, search_length = G.route()
    ant_path, ant_length = C(100)
    print(G.edges.value)
    print(ant_path, search_path)
    print(ant_length, search_length)


