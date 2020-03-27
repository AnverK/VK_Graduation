import numpy as np
from scipy.stats import gamma, ttest_1samp

from causality.pc.independence.utils import kernel_matrix


def kernel_based_indepence(x, y, eigv_samples=1000, approximate=True):
    n_samples, _ = x.shape
    kx = kernel_matrix(x)
    ky = kernel_matrix(y)

    if approximate:
        # kx = pairwise_kernels(x, metric='rbf', gamma=np.median(pdist(x)))
        # ky = pairwise_kernels(y, metric='rbf', gamma=np.median(pdist(y)))
        # h = np.identity(n_samples) - np.full((n_samples, n_samples), 1 / n_samples)
        # cx = h @ kx @ h
        # cy = h @ ky @ h
        mean_appr = np.trace(kx) * np.trace(ky) / n_samples
        var_appr = 2 * (n_samples - 4) * (n_samples - 5) * np.linalg.norm(kx) * np.linalg.norm(ky) / (n_samples ** 4)
        k_appr = mean_appr * mean_appr / var_appr
        theta_appr = var_appr / mean_appr
        Sta = np.trace(kx @ ky)
        return gamma.cdf(Sta, a=k_appr, scale=theta_appr)

    eig_x = np.linalg.eigvalsh(kx)
    eig_y = np.linalg.eigvalsh(ky)

    z = np.random.chisquare(1, (n_samples * n_samples, eigv_samples))
    eigs = np.outer(eig_x, eig_y).flatten()
    t_samples = np.dot(eigs, z) / (n_samples * n_samples)

    actual = 1 / n_samples * np.trace(kx @ ky)
    t, p_value = ttest_1samp(t_samples, actual)
    print(t, np.mean(t_samples), actual)
    if t < 0:
        p_value = 1 - p_value / 2
    else:
        p_value = p_value / 2

    return p_value
