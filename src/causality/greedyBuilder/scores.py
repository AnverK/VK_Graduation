from scipy.spatial import KDTree
from scipy.special import digamma
from sklearn.neighbors import KDTree
import numpy as np
import networkx as nx
from math import log


def avg_digamma(points, dvec):
    tree = KDTree(points, metric='chebyshev')
    dvec = dvec - 1e-15
    num_points = tree.query_radius(points, dvec, count_only=True)
    return np.mean(digamma(num_points))


def mi(x, y, k=3):
    points = [x, y]
    points = np.hstack(points)
    tree = KDTree(points, metric='chebyshev')
    dvec = tree.query(points, k=k + 1)[0][:, k]
    a, b, c, d = avg_digamma(x, dvec), avg_digamma(y, dvec), digamma(k), digamma(len(x))
    return -a - b + c + d


def compute_likelihood(data, graph):
    data += np.random.random_sample(data.shape) * 1e-10
    result = np.zeros(np.shape(data)[1])
    indep = 0
    for v in nx.nodes(graph):
        parents = list(graph.predecessors(v))
        y = data[:, v].reshape(-1, 1)
        if len(parents) == 0:
            indep += 1
        else:
            x = data[:, parents]
            result[v] = mi(x, y, 3)
    M = len(data)
    penalty = log(M) / 2 * indep
    return result, penalty
