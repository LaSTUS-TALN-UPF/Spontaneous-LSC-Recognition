import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 61
self_link = [(i, i) for i in range(num_node)]

inward = [
    (0,0),(0,1),(0,2),(1,3),(2, 4),(0,5),(0,6),(5,7),(6,8),(7,9),(8,10), (9, 11), (10, 12), (9,13), (10, 14), (9,15), (10, 16) ,(5,17), (6,18),
    (9, 19), (19, 20), (20, 21), (21, 22), (22, 23), (19, 24), (24, 25), (25, 26), (26, 27), (19, 28), (28, 29), (29, 30), (30, 31), (19, 32), (32, 33), (33, 34), (34, 35), (19, 36), (36, 37), (37, 38), (38, 39),
    (10, 40), (40, 41), (41, 42), (42, 43), (43, 44), (40, 45), (45, 46), (46, 47), (47, 48), (40, 49), (49, 50), (50, 51), (51, 52), (40, 53), (53, 54), (54, 55), (55, 56), (40, 57), (57, 58), (58, 59), (59, 60)
    ]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
