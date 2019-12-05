import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

from src.utils import discretize_metric, get_positive_threshold, get_negative_threshold


class MetricLinearRegression(BaseEstimator):
    def __init__(self, alpha=1, beta=1, aa_p_value=20, painted_p_value=99, pos_threshold=None, neg_threshold=None,
                 coef=None,
                 l2_reg=0):
        self.alpha = alpha
        self.beta = beta
        self.aa_p_value = aa_p_value
        self.painted_p_value = painted_p_value
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.coef = coef
        self.l2_reg = l2_reg

    def __objective_function(self, w, X_e, X_aa, X_u, y_e):
        e_term = np.dot(np.dot(X_e, w), y_e)
        aa_term = np.sum(np.absolute(np.dot(X_aa, w)))
        # u_term = np.sum(np.abs(np.dot(X_u, w)))
        return -e_term / len(X_e) \
               + self.alpha * aa_term / len(X_aa)
        # - self.beta * u_term / len(X_u) \
        # - self.l2_reg * (np.linalg.norm(w) ** 2)

    def __jacobian(self, w, X_e, X_aa, X_u, y_e):
        e_term = np.dot(y_e, X_e)
        aa_term = np.dot(X_aa.T, np.sign(np.dot(X_aa, w)))
        # u_term = np.dot(np.sign(np.dot(X_u, w)), X_u)
        return -e_term / len(X_e) \
               + self.alpha * aa_term / len(X_aa)
        # - self.beta * u_term / len(X_u) \
        # - 2 * self.l2_reg * w

    def fit(self, X, y):
        if self.pos_threshold is None:
            self.pos_threshold = get_positive_threshold(y, self.painted_p_value)
        if self.neg_threshold is None:
            self.neg_threshold = get_negative_threshold(y, self.painted_p_value)

        n_samples, n_features = X.shape
        painted_experiments = np.argwhere(y[:, 1] > self.painted_p_value)
        uncertain_experiments = np.argwhere(y[:, 1] <= self.painted_p_value)
        aa_experiments = np.argwhere(y[:, 1] < self.aa_p_value)

        X_painted = X[painted_experiments].reshape(len(painted_experiments), n_features)
        y_painted = y[painted_experiments, 0].flatten()
        y_painted = discretize_metric(y_painted, self.pos_threshold, self.neg_threshold)

        X_aa = X[aa_experiments].reshape(len(aa_experiments), n_features)

        X_uncertain = X[uncertain_experiments].reshape(len(uncertain_experiments), n_features)

        if self.coef is None:
            self.coef = np.random.rand(n_features)

        result = minimize(self.__objective_function, self.coef, (X_painted, X_aa, X_uncertain, y_painted),
                          method='BFGS')
        self.coef = result.x
        print(self.coef)
        print(self.__objective_function(self.coef, X_painted, X_aa, X_uncertain, y_painted))
        print(result.message)

    def predict(self, X):
        dots = np.dot(self.coef, X.T)
        return discretize_metric(dots, self.pos_threshold, self.neg_threshold)
