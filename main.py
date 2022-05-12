from graph import *
from aco import *



if __name__ == '__main__':
    G = Germany()
    AS = AntSystem(20, G, alpha=3.0, beta=3.0, rho=0.1)
    ant_path, ant_length, pheromone = AS(100)#, visualization=lambda: G.visualize())
    G.visualize()

    G = Germany()
    ACS = AntColonySystem(20, G, alpha=2.0, beta=3.0, rho=0.1, phi=0.3, q=0.1, tau=0.0)
    ant_path, ant_length, pheromone = ACS(100)
    G.visualize()
