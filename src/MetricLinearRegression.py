import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from src.utils import discretisize_metric, find_index_of_nearest


class MetricLinearRegression(BaseEstimator):
    def __init__(self, l2_reg=1, beta=1, aa_p_value=20, painted_p_value=99, threshold=None, coef=None):
        self.alpha = beta
        self.coef = coef
        self.aa_p_value = aa_p_value
        self.painted_p_value = painted_p_value
        self.l2_reg = l2_reg
        self.threshold = threshold

    def __objective_function(self, w, X_e, X_aa, y_e):
        e_term = sum([np.dot(w, x) * y for x, y in zip(X_e, y_e)])
        aa_term = sum([abs(np.dot(w, x)) for x in X_aa])
        return -e_term / len(X_e) - self.alpha * aa_term / len(aa_term)

    def fit(self, X, y):
        if self.threshold is None:
            self.threshold = max(np.abs(y[np.argwhere(y[:, 1] < self.painted_p_value), 0]))

        n_samples, n_features = X.shape
        painted_experiments = np.argwhere(y[:, 1] > self.painted_p_value)
        aa_experiments = np.argwhere(y[:, 1] < self.aa_p_value)

        X_painted = X[painted_experiments].reshape(len(painted_experiments), n_features)
        X_aa = X[aa_experiments].reshape(len(aa_experiments), n_features)
        y_painted = y[painted_experiments, 0].flatten()
        y_painted = discretisize_metric(y_painted, self.threshold)

        if self.coef is None:
            self.coef = np.random.rand(n_features)

        result = minimize(self.__objective_function, self.coef, (X_painted, X_aa, y_painted),
                          method='L-BFGS-B')
        self.coef = result.x
        # print(result.success)

    def predict(self, X):
        dots = np.dot(self.coef, X.T)
        return discretisize_metric(dots, self.threshold)
