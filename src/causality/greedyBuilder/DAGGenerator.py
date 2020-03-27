from math import sqrt
import numpy as np


class DAGGenerator:
    def __init__(self, n_verts, max_degree='auto', minimal=False):
        self.n_verts = n_verts
        self.minimal = minimal
        if max_degree == 'auto':
            self.max_degree = int(sqrt(n_verts))
        else:
            self.max_degree = max_degree

    def generate(self):
        edges = np.zeros((self.n_verts, self.n_verts), dtype=int)
        verts_order = np.random.permutation(self.n_verts)
        degrees = np.random.randint(1, self.max_degree, size=self.n_verts)
        for i, v in enumerate(verts_order[:-1]):
            children = np.random.choice(verts_order[i + 1:], min(degrees[i], len(verts_order) - 1 - i), replace=False)
            edges[v][children] = 1
        return edges
