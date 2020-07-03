import math

import numpy as np
from scipy.stats import norm


class GaussConditionalIndepTest:

    def __init__(self, corr):
        self.correlation_matrix = corr
        self.n = len(corr)

    def gauss_ci_test(self, i, j, ks):
        C = self.correlation_matrix
        cut_at = 0.9999999
        if len(ks) == 0:
            r = C[i, j]
        elif len(ks) == 1:
            k = ks[0]
            r = (C[i, j] - C[i, k] * C[j, k]) / math.sqrt((1 - C[j, k] ** 2) * (1 - C[i, k] ** 2))
        else:
            m = C[np.ix_([i] + [j] + ks, [i] + [j] + ks)]
            PM = np.linalg.pinv(m)
            r = -PM[0, 1] / math.sqrt(PM[0, 0] * PM[1, 1])
        r = min(cut_at, r)
        res = math.sqrt(self.n - len(ks) - 3) * .5 * abs(math.log1p((2 * r) / (1 - r)))
        return 2 * (norm.sf(res))
