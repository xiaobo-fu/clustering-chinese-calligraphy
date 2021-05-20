import numpy as np
import networkx as nx
import functions
import pygame


# get the range of a list
def list_range(val_list):
    min_val = min(val_list)
    max_val = max(val_list)
    return [min_val, max_val]


# build networkx graph from graph dictionary
def build_nx_graph(graphDict):
    G = nx.from_dict_of_lists(graphDict)
    for e in G.edges():
        vi = e[0]
        vj = e[1]
        G[vi][vj]['distance'] = 1
    return G


# create a line graph for SOM
def create_cyclic_graph(k):
    G = {}
    # vertices
    V = np.arange(k)
    # assign edges
    for i in V[0:-1]:
        leftNode = V[i]
        rightNode = V[i+1]
        G[i] = [leftNode, rightNode]
    return G


# get the R range, ie, the 'box'
def get_r(X, w):
    R = []
    for i in range(w):
        # get the min and max of map columns, ie, R in each dimension
        R.append(list_range(X[:, i]))
    return np.array(R)


# generate random weights for use in SOM
def random_weight(X, k):
    # width of the map, ie, the dimension of the data points
    w = len(X[0])
    # get R size, ie, the 'box'
    R = get_r(X, w)
    # k weights with length of w
    W = np.random.rand(k, w)
    # scale all the weights according to R
    RMax = R[:, 1]
    RMin = R[:, 0]
    W = RMin + W * (RMax - RMin)
    return W


# create the topological distance matrix
def create_d(G, k):
    D = np.zeros((k, k))

    # convert G to networkx graph
    nxG = build_nx_graph(G)

    for i in range(k):
        for j in range(i + 1, k):
            path = nx.dijkstra_path(nxG, i, j, weight='distance')
            dij = len(path) - 1
            D[i, j] = dij
            D[j, i] = D[i, j]
    return D


# get winner neuron index i
def get_winner(x, W, k):
    # a list distances
    distances = []

    # append all k distances
    for i in range(k):
        w = W[i]
        dwx = np.linalg.norm(w - x)
        distances.append(dwx)

    return np.argmin(distances)


# update all the weights
def update_all_weights(iWinnder, x, D, W, k, t, tMax):
    learningRate = 1 - (t / tMax)
    topologicalAdaptionRate = np.exp(- t / tMax)
    for j in range(k):
        W[j] = W[j] + (x - W[j]) * (learningRate * np.power(np.e, (- D[iWinnder, j] / (2 * topologicalAdaptionRate))))
    return W


def self_organizing_map(X, k, tMax):
    # create cyclic graph G
    G = create_cyclic_graph(k)

    # initialize random vertex weight
    W = random_weight(X, k)

    # create topological distance matrix D
    D = create_d(G, k)

    # som iteration
    for t in range(tMax):
        # get random point x in X
        idx = np.random.randint(X.shape[0])
        x = X[idx]

        # get winner neuron index iWinnder
        iWinnder = get_winner(x, W, k)

        # update all the weights
        W = update_all_weights(iWinnder, x, D, W, k, t, tMax)

        # visualization
        # clear surface
        win.fill((255, 255, 255))
        # draw the character image a background
        win.blit(character_img, (0, 0))
        # draw vertices
        for w in W:
            pygame.draw.circle(win, (255, 0, 0), (w[0] * scaler, w[1] * scaler), 3)
        # draw edges
        for i in range(1, k):
            v1 = i
            v2 = i + 1
            p1 = np.array([W[-v1][0], W[-v1][1]])
            p2 = np.array([W[-v2][0], W[-v2][1]])
            pygame.draw.line(win, (255, 0, 0), p1 * scaler, p2 * scaler, 2)

        pygame.display.update()
        clock.tick(1000)
    pygame.image.save(win, f"result/Self Organizing Map 0{image_number}.jpg")
    return G, W


if __name__ == "__main__":
    # visualize using pygame
    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # read image data
    image_number = 1
    url = f'jpg/256/{image_number}.JPG'
    character_map, character_points = functions.image_to_arrays(url)
    character_img = pygame.image.load(url)

    scaler = 2
    width = len(character_map[0]) * scaler
    height = len(character_map) * scaler
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Self Organizing Map')

    # a background surface to display the character image
    win.fill((255, 255, 255))

    character_img = pygame.transform.scale(character_img, (width, height))
    win.blit(character_img, (0, 0))
    pygame.display.update()

    # set total nodes k and iteration times t
    k = 100
    t = 1000

    image_number = 0
    # pygame loop
    run = True
    while run:
        # quit if end
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # loop all the images
        image_number += 1
        url = f'jpg/256/{image_number}.JPG'
        character_map, character_points = functions.image_to_arrays(url)
        character_img = pygame.image.load(url)
        character_img = pygame.transform.scale(character_img, (width, height))

        self_organizing_map(character_points, k, t)

        if image_number == 8:
            run = False

