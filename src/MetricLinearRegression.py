import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

from src.utils import discretize_metric, get_positive_threshold, get_negative_threshold


class MetricLinearRegression(BaseEstimator):
    def __init__(self, l2_reg=1, beta=1, critical_p_value=99, pos_threshold=None, neg_threshold=None, coef=None):
        self.beta = beta
        self.coef = coef
        self.critical_p_value = critical_p_value
        self.l2_reg = l2_reg
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def __objective_function(self, w, X_e, X_u, y_e):
        e_term = sum([np.dot(w, x) * y for x, y in zip(X_e, y_e)])
        u_term = sum([abs(np.dot(w, x)) for x in X_u])
        return -e_term / len(X_e) - self.beta * u_term / len(X_u) + self.l2_reg * (np.linalg.norm(w) ** 2)

    def fit(self, X, y):
        if self.pos_threshold is None:
            self.pos_threshold = get_positive_threshold(y, self.critical_p_value)
        if self.neg_threshold is None:
            self.neg_threshold = get_negative_threshold(y, self.critical_p_value)

        n_samples, n_features = X.shape
        painted_experiments = np.argwhere(y[:, 1] > self.critical_p_value)
        uncertain_experiments = np.argwhere(y[:, 1] <= self.critical_p_value)

        X_painted = X[painted_experiments].reshape(len(painted_experiments), n_features)
        X_uncertain = X[uncertain_experiments].reshape(len(uncertain_experiments), n_features)
        y_painted = y[painted_experiments, 0].flatten()
        y_painted = discretize_metric(y_painted, self.pos_threshold, self.neg_threshold)

        if self.coef is None:
            self.coef = np.random.rand(n_features)

        result = minimize(self.__objective_function, self.coef, (X_painted, X_uncertain, y_painted),
                          method='L-BFGS-B')
        self.coef = result.x
        print(self.coef)

    def predict(self, X):
        dots = np.dot(self.coef, X.T)
        return discretize_metric(dots, self.pos_threshold, self.neg_threshold)
