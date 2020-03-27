from scipy.spatial.distance import pdist, squareform, cdist
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from sklearn.neighbors import NearestNeighbors


def mutual_info_pairwise(x, y):
    return mutual_info_regression(x.reshape(-1, 1), y)[0]


def mutual_info(X, y):
    if X.shape[1] == 0:
        return 0
    # if X.shape[1] == 1:
    #     return mutual_info_pairwise(X, y)
    return compute_mi_cc_2(X, y.reshape(-1, 1), 3)


def compute_mi_cc_1(x, y, n_neighbors):
    n_samples = len(y)

    dist_x = pdist(x)
    dist_y = pdist(y)
    dist_z = squareform(np.maximum(dist_x, dist_y), force='tomatrix', checks=False)

    nn = NearestNeighbors(metric='precomputed', n_neighbors=n_neighbors)
    nn.fit(dist_z)
    nn.kneighbors()
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    nn = NearestNeighbors(algorithm='brute', n_neighbors=n_neighbors)
    nn.fit(x)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    nx = np.array([i.size for i in ind])

    nn.fit(y)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    ny = np.array([i.size for i in ind])

    mi = digamma(n_samples) + digamma(n_neighbors) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))

    return max(0, mi)


def compute_mi_cc_2(x, y, n_neighbors):
    n_samples = len(y)
    sample_inds = np.arange(n_samples).reshape(-1, 1)

    dist_x = squareform(pdist(x), force='tomatrix', checks=False)
    dist_y = squareform(pdist(y), force='tomatrix', checks=False)
    dist_z = np.maximum(dist_x, dist_y)

    nn = NearestNeighbors(metric='precomputed', n_neighbors=n_neighbors, n_jobs=3)
    nn.fit(dist_z)
    nn.kneighbors()
    ind_z = nn.kneighbors(return_distance=False)

    rx = np.max(dist_x[sample_inds, ind_z], axis=1).reshape(-1, 1)
    nx = np.count_nonzero(dist_x < rx, axis=1)

    ry = np.max(dist_y[sample_inds, ind_z], axis=1).reshape(-1, 1)
    ny = np.count_nonzero(dist_y < ry, axis=1)

    mi = digamma(n_samples) + digamma(n_neighbors) - np.mean(digamma(nx)) - np.mean(digamma(ny)) - 1 / n_neighbors

    return max(0, mi)

# x1 = np.random.randn(100, 10)
# x2 = np.random.randn(100, 1)
# x = np.hstack((x1, x2))
# y = np.sum(x1, axis=1).reshape(-1, 1)
# y = np.random.randn(100, 1)
# y = np.random.normal(-2, 1, 1000)
# print(compute_mi_cc_1(x1, y, 3))
# print(compute_mi_cc_2(x1, y, 3))
# print(mutual_info_regression(x, y))
