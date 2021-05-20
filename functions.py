from PIL import Image
import numpy as np


def image_to_arrays(url, threshold=128):
    img = np.array(Image.open(url))
    bmp_arr = np.zeros((len(img), len(img[0])), dtype=np.int8)
    points_arr = []
    for y, row in enumerate(img):
        for x, point in enumerate(row):
            if np.sum(point) / 3 <= threshold:
                bmp_arr[y, x] = 1
                points_arr.append([x, y, 0])
    return bmp_arr, np.array(points_arr)


# from the txt, generate map, graph G, vertices to cords V2C, and cords to vertices C2V, adjacency matrix A, and degree matrix D
def generate_map(file):
    f = file

    # a 2D array map
    map = []

    # get rows and columns from f, delete space and newline
    for r in f:
        mapRow = []
        for c in r:
            if c != ' ' and c != '\n':
                mapRow.append(c)
        map.append(mapRow)

    # flip map upside down, so (0, 0) is the lower left corner
    flippedMap = np.flipud(map)

    # width and height of the map
    w = len(map[0])
    h = len(map)

    # non-wall vertices dict, and coordinates dict
    C2V = {}
    V2C = {}

    # check all the fields in the flipped map, get all the non-wall fields
    # node counter
    n = 0
    for y in range(h):
        for x in range(w):
            # store current field cords if not wall
            if flippedMap[y, x] == 1:
                V2C[n] = (x, y)
                C2V[(x, y)] = n
                n += 1

    # graph dict, adjacency matrix with zeros, degree matrix with zeros
    G = {}

    vn = len(C2V)

    A = np.zeros((vn, vn))
    D = np.zeros((vn, vn))

    # create graph edges
    # check all vertices in V2C
    for i in V2C:
        # a list to store current field's edges and weight
        edges = []

        # degree of a vertex
        d = 0

        # cords of current field
        x, y = V2C.get(i)

        # cords around the current field
        aroundCourds = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        for aroundCourd in aroundCourds:
            # check if the around cords is in the C2V dict, ie, if it's non-wall, then append to edges and A
            if aroundCourd in C2V:
                aroundVetex = C2V.get(aroundCourd)
                edges.append(aroundVetex)

                # assign the adjacency matrix
                A[i, aroundVetex] = 1
                A[aroundVetex, i] = 1

                # degree + 1
                d += 1

            # assign degree matrix
            D[i, i] = d
        G[i] = edges
    return map, G, C2V, V2C, A, D
