import numpy as np
import pygame
import functions


# generate n random weights
def random_weight(C, n):
    # width of and height of the data
    w = len(C[0])
    h = len(C)
    R = [w, h]

    # n random sample with length of w
    W = np.random.rand(n, 2)

    # scale by R
    W = W * R
    return W


def get_distances_between_V_and_x(V, x):
    D = []
    for i, w in V.items():
        # distance between x and v
        d = np.linalg.norm(x - w)
        D.append(d)
    return D


def get_neighbours(v, E):
    neighbour = []
    # get vertex v's edges
    for e in E[v]:
        # get the neighbour vertices in the edge
        for n in e:
            # if the vertex is not s self
            if n != v:
                neighbour.append(n)
    return neighbour


def get_edge(v1, v2):
    return (v1, v2) if v2 > v1 else (v2, v1)


# draw the map with grid
def draw_grid(map, scaler):
    win.fill((255, 255, 255))
    width = len(map[0]) * scaler
    height = len(map) * scaler
    for c in range(len(map[0])):
        pygame.draw.line(win, (238, 238, 238), (c * scaler, 0), (c * scaler, height))
    for r in range(len(map)):
        pygame.draw.line(win, (238, 238, 238), (0, r * scaler,), (width, r * scaler,))
        for c in range(len(map[0])):
            if map[r][c] == '1':
                pygame.draw.rect(win, 0, (c * scaler, r * scaler, scaler, scaler))


def draw_character(map, scaler):
    for i, r in enumerate(map):
        for j, c in enumerate(r):
            if c != 0:
                pygame.draw.rect(win, 0, (j * scaler, i * scaler, scaler, scaler))


def neural_gas(X, tmax=10000, deltaWinner=0.1, deltaNeighbour=0.01, aMax=10, lam=20, N=200, alpha=0.9, beta=0.9):
    # initialization
    # create labeled graph G
    W = random_weight(X, 2)
    V = {0: W[0], 1: W[1]}
    E = {0: [(0, 1)], 1: [(0, 1)]}
    err = {0: 0, 1: 0}
    age = {(0, 1): 0}

    # get point data in X
    C = []
    for i, r in enumerate(X):
        for j, p in enumerate(r):
            if p != 0:
                C.append((j, i))

    # start the very long iteration
    for t in range(tmax):
        # sample a point in C
        x = C[np.random.choice(len(C))]

        # get a list of distances of points in V to x
        D = get_distances_between_V_and_x(V, x)

        # determine the nodes s, r in V closest to x
        # find the index of two smallest d
        sIndex, rIndex = np.argsort(D)[: 2]

        # because some vertices will be deleted, the sIndex, rIndex is not corresponding to the vertices keys but to
        # its sequence, so we need to turn the keys to a list first
        VKeys = list(V.keys())
        s = VKeys[sIndex]
        r = VKeys[rIndex]

        # update the error of the winner node s
        err[s] = err[s] + D[sIndex]

        # move the winner node s towards x
        V[s] = V[s] + deltaWinner * (x - V[s])

        # move all topological neighbors of s towards x
        for n in get_neighbours(s, E):
            V[n] = V[n] + deltaNeighbour * (x - V[n])

        # increment the age of all edges incident to winner s
        for e in E[s]:
            age[e] += 1

        # create / set the age of edge between winner s and runner up r to 0
        newEdge = get_edge(s, r)
        if newEdge not in E[s]:
            E[s].append(newEdge)
            E[r].append(newEdge)
        age[newEdge] = 0

        # remove edges older than aMax
        # a list of old edges
        oldEdges = []
        for e, a in age.items():
            if a > aMax:
                # remove from E
                v1 = e[0]
                v2 = e[1]
                oldEdge = get_edge(v1, v2)
                E[v1].remove(oldEdge)
                E[v2].remove(oldEdge)

                oldEdges.append(oldEdge)

                # if produces isolated nodes, remove from V
                if not E[v1]:
                    V.pop(v1)
                if not E[v2]:
                    V.pop(v2)

        for e in oldEdges:
            # remove from age dict
            age.pop(e)

        # create new node every lambda round
        if (t+1) % lam == 0 and len(V) <= N:
            # find the node with the largest error
            errKey = list(err.keys())
            errValue = list(err.values())
            errIndex = np.argmax(errValue)
            vMaxErr = errKey[errIndex]

            # find vMaxErr's neighbour with largest error
            # first find all the neighbours
            neighbour = get_neighbours(vMaxErr, E)

            # get neighbour error
            neighbourIndices = []
            neighbourErr = []
            for n in neighbour:
                neighbourIndices.append(n)
                neighbourErr.append(err[n])

            errIndex = np.argmax(neighbourErr)
            vNeighbourMaxErr = neighbourIndices[errIndex]

            # update err of the two vertices
            err[vMaxErr] = alpha * err[vMaxErr]
            err[vNeighbourMaxErr] = alpha * err[vNeighbourMaxErr]

            # find the largest keys, +1 as new node
            newNode = np.max(list(V.keys())) + 1

            err[newNode] = err[vMaxErr]
            V[newNode] = (V[vMaxErr] + V[vNeighbourMaxErr]) / 2

            # create edges between the new node and the two with largest err
            newEdge = get_edge(newNode, vMaxErr)
            E[newNode] = [newEdge]
            E[vMaxErr].append(newEdge)
            age[newEdge] = 0

            newEdge = get_edge(newNode, vNeighbourMaxErr)
            E[newNode].append(newEdge)
            E[vNeighbourMaxErr].append(newEdge)
            age[newEdge] = 0

            # remove edge between the two with largest err
            oldEdge = get_edge(vMaxErr, vNeighbourMaxErr)
            E[vMaxErr].remove(oldEdge)
            E[vNeighbourMaxErr].remove(oldEdge)
            age.pop(oldEdge)

            # decrease the error value of all nodes
            for i in err:
                err[i] = beta * err[i]

        # visualization
        # clear surface
        win.fill((255, 255, 255))
        # draw the character image a background
        win.blit(character_img, (0, 0))
        # draw vertices
        for v in V:
            pygame.draw.circle(win, (255, 0, 0), V[v] * scaler, 3)
        # draw edges
        for e in age:
            v1 = e[0]
            v2 = e[1]
            p1 = V[v1]
            p2 = V[v2]
            pygame.draw.line(win, (255, 0, 0), p1 * scaler, p2 * scaler, 2)

        pygame.display.update()
        pygame.image.save(win, f"result/Neural Gas 0{image_number}.jpg")
        clock.tick(120)


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
    pygame.display.set_caption('Neural Gas')

    # a background surface to display the character image
    win.fill((255, 255, 255))

    character_img = pygame.transform.scale(character_img, (width, height))
    win.blit(character_img, (0, 0))
    pygame.display.update()

    image_number = 0
    # pygame loop
    run = True
    while run:
        # quit if end
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # pygame.time.wait(10000)

        # loop all the images
        image_number += 1

        url = f'jpg/256/{image_number}.JPG'
        character_map, character_points = functions.image_to_arrays(url)
        character_img = pygame.image.load(url)
        character_img = pygame.transform.scale(character_img, (width, height))

        neural_gas(character_map, tmax=10000, deltaWinner=0.05, deltaNeighbour=0.01, aMax=20, lam=20, N=200, alpha=0.9,
                   beta=0.9)

        if image_number == 8:
            run = False




