from graph import *
from aco import AntColony


def visualize(pheromone_levels, path):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Route')

    # Mark all cities with a dot
    coords = np.array([node.coordinates for node in ant_path])
    ax.scatter(coords[:, 0], coords[:, 1])

    # Normalize the pheromone levels
    max_phero = np.max(pheromone[pheromone != 1.0])
    min_phero = np.min(pheromone[pheromone != 1.0])

    for node1 in ant_path:
        for node2 in ant_path:

            # Draw the connection of the two nodes based on the pheromone concentration
            if not node1 == node2:

                x_values = [node1.coordinates[0], node2.coordinates[0]]
                y_values = [node1.coordinates[1], node2.coordinates[1]]

                # Normalize pheromones for better visibility
                normed_pheromone = (pheromone_levels[node1.name, node2.name] - min_phero) / (max_phero - min_phero)

                # Draw the connection based on determined pheromone level
                ax.plot(x_values, y_values, '-', color=[0, 0, 0, normed_pheromone])

    ax.imshow(plt.imread("osm_germany.png"))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    city_coords_dict = {'Osnabrück': (235, 234),
                        'Hamburg': (324, 137),
                        'Hanover': (312, 226),
                        'Frankfurt': (264, 391),
                        'Munich': (396, 528),
                        'Berlin': (478, 215),
                        'Leipzig': (432, 302),
                        'Düsseldorf': (175, 310),
                        'Kassel': (300, 300),
                        'Wolfsburg': (360, 215),
                        'Münster': (200, 275)}

    # coordinates for osna, hamburg, hanover, frankfurt, munich, berlin and leipzig, kassel, Düsseldorf
    coordinates = list(city_coords_dict.values())
    G = TSP(coordinates=coordinates)
    C = AntColony(20, G, alpha=1, beta=2.0, rho=0.2)
    print(G[1, 2] == G[2, 1])
    #search_path, search_length = G.route()
    ant_path, ant_length, pheromone = C(1000)

    # print('Pheromone: ', np.max(pheromone[pheromone != 1.0]))

    visualize(pheromone, ant_path)