import numpy as np
from scipy.stats import ttest_1samp, gamma
from causality.pc.independence.utils import kernel_matrix


def kernel_based_conditional_independence(x, y, Z, eps=1e-3, approximate=True, threshold=1e-6, bs_iters=5e3):
    # normalize the data??
    n_samples, n_features = Z.shape
    xz = np.append(Z, x, axis=1)
    Kx = kernel_matrix(xz)
    Ky = kernel_matrix(y)
    Kz = kernel_matrix(Z)

    P1 = eps * np.linalg.inv(Kz + eps * np.identity(n_samples))
    Kxz = P1 @ Kx @ P1.T
    Kyz = P1 @ Ky @ P1.T

    Sta = np.trace(Kxz @ Kyz)

    eig_Kxz, eivx = np.linalg.eigh((Kxz + Kxz.T) / 2)
    eig_Kyz, eivy = np.linalg.eigh((Kyz + Kyz.T) / 2)

    IIx = np.argwhere(eig_Kxz > eig_Kxz[-1] * threshold).flatten()
    IIy = np.argwhere(eig_Kyz > eig_Kyz[-1] * threshold).flatten()
    eig_Kxz = eig_Kxz[IIx]
    eig_Kyz = eig_Kyz[IIy]
    eivx = eivx[:, IIx]
    eivy = eivy[:, IIy]

    eiv_prodx = eivx @ np.diag(np.sqrt(eig_Kxz))
    eiv_prody = eivy @ np.diag(np.sqrt(eig_Kyz))

    uu = np.array(list(map(np.outer, eiv_prodx, eiv_prody)))
    uu = uu.reshape(uu.shape[0], uu.shape[1] * uu.shape[2])

    assert uu.shape[0] == n_samples
    if uu.shape[1] > uu.shape[0]:
        uu_prod = uu @ uu.T
    else:
        uu_prod = uu.T @ uu

    if approximate:
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod @ uu_prod)
        k_appr = mean_appr * mean_appr / var_appr
        theta_appr = var_appr / mean_appr
        return gamma.cdf(Sta, a=k_appr, scale=theta_appr)

    eig_uu = np.linalg.eigvalsh(uu_prod)
    II_f = np.argwhere(eig_uu > eig_uu[-1] * threshold).flatten()
    eig_uu = eig_uu[II_f]
    bs_iters = int(bs_iters)
    f_rand1 = np.random.chisquare(1, (len(eig_uu), bs_iters))
    Null_dstr = eig_uu.T @ f_rand1
    t, p_value = ttest_1samp(Null_dstr, Sta)
    if t < 0:
        p_value = 1 - p_value / 2
    else:
        p_value = p_value / 2

    return p_value
