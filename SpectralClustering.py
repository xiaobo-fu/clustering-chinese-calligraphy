import numpy as np
import pygame
from sklearn.cluster import KMeans
import functions


# draw the node
def draw_rect(node, scaler, m, V2C, color=(220, 0, 0)):
    x, flippedY = V2C[node]
    y = len(m) - flippedY - 1
    pygame.draw.rect(win, color, (x * scaler + .5, y * scaler + .5, scaler - 1, scaler - 1))


if __name__ == "__main__":
    # read image data
    image_number = 3
    url = f'jpg/064/{image_number}.jpg'
    character_map, character_points = functions.image_to_arrays(url)

    k = 2

    # from the txt, generate map, graph G, vertices' cords C, adjacency matrix A, and degree matrix D
    m, G, C2V, V2C, A, D = functions.generate_map(character_map)

    # Laplacian matrix L
    L = D - A

    # find the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)

    # sort these based on the eigenvalues
    vecs = vecs[:, np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    # kmeans on first three vectors with nonzero eigenvalues
    kmeans = KMeans(n_clusters=k)

    kmeans.fit(vecs[:, 1:k])
    result = kmeans.labels_

    # visualize using pygame
    # setting up pygame window
    pygame.init()
    scaler = 8
    width = len(m[0]) * scaler
    height = len(m) * scaler
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Path Planning')
    win.fill((255, 255, 255))
    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]

    # pygame loop
    run = True
    while run:
        # quit if end
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        for i in V2C:
            color = colors[result[i]]
            draw_rect(i, scaler, m, V2C, color)

        pygame.display.update()
        pygame.image.save(win, f"result/Spectral Clustering 0{image_number} k={k}.jpg")

        pygame.time.wait(10000)
        run = False

