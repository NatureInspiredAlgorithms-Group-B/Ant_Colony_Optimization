from graph import *
from aco import *



if __name__ == '__main__':
    #G = Germany()
    #AS = AntSystem(20, G, alpha=3.0, beta=3.0, rho=0.1)
    #ant_path, ant_length, pheromone = AS(100)#, visualization=lambda: G.visualize())
    #G.visualize()
    #
    #G = Germany()
    #print('pheromone' in G._edge_values.keys())
    #ACS = AntColonySystem(20, G, alpha=2.0, beta=3.0, rho=0.1, phi=0.3, q=0.1, tau=0.0)
    #ant_path, ant_length, pheromone = ACS(100)
    #print('pheromone' in G._edge_values.keys())
    #G.visualize()

    
    G = AntWorld('world_1.ant')
    AS = AntSystem(10, G, alpha=2.0, beta=3.0, rho=0.1,
                    valid=lambda a: any(edge.heuristic != 1 for edge in a.path_edges))
    

    def visualization():
        pheromones = G.visualize(active=False)
        ants = np.zeros_like(pheromones)
        for ant in AS:
            i, j = ant.node.name.split('|')
            i, j = int(i), int(j)
            ants[i, j] = 1.0
            print(i, j)
        img = np.ones(pheromones.shape + (3,))
        img *= pheromones[..., None]
        img /= np.max(img)
        img[ants==1,:] = 0
        plt.imshow(img)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)
        print(pheromones)


    AS(10, visualization=visualization)
    exit()
