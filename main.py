from graph import *
from aco import AntColony


if __name__ == '__main__':
    # coordinates for osna, hamburg, hanover, frankfurt, munich, berlin and leipzig
    coordinates = [(235, 234),(324, 137), (312, 226), (264, 391), (396, 528), (478, 215), (432, 302)]
    G = TSP(coordinates=coordinates)
    C = AntColony(10, G, alpha=1, beta=1, rho=0.1)
    print(G[1, 2] == G[2, 1])
    search_path, search_length = G.route()
    ant_path, ant_length = C(100)
    print(G.edges.value)
    print(ant_path, search_path)
    print(ant_length, search_length)
    Graph().visualize(coordinates)
