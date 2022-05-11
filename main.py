from graph import *
from aco import AntColony



if __name__ == '__main__':
    G = Germany()
<<<<<<< HEAD
    C = AntColony(50, G, alpha=3.0, beta=3.0, rho=0.1)
    ant_path, ant_length, pheromone = C(10, visualization=lambda: G.visualize())
    #G.visualize()
=======
    C = AntColony(20, G, alpha=3.0, beta=3.0, rho=0.1)
    ant_path, ant_length, pheromone = C(100)
    G.visualize()
>>>>>>> 9fdee09 (add the notebook)
