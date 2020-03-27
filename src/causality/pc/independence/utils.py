import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_kernels


def kernel_matrix(x):
    n_samples, _ = x.shape
    h = np.identity(n_samples) - np.full((n_samples, n_samples), 1 / n_samples)
    kx = pairwise_kernels(x, metric='rbf', gamma=np.median(pdist(x)))
    return h @ kx @ h
