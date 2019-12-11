import numpy as np
from sklearn.base import BaseEstimator

from src.utils import *


class InstrumentalVariable(BaseEstimator):
    def __init__(self, critical_p_value=95, l2_reg=None, coef=None):
        self.critical_p_value = critical_p_value
        self.coef = coef
        self.l2_reg = l2_reg

    def __handle_features(self, X):
        return np.array(list(map(lambda metric: apply_l0_regularization(metric, self.critical_p_value), X)))[:, :, 0]

    def fit(self, X, y):
        X = self.__handle_features(X)
        if self.l2_reg is None:
            self.coef = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            n_samples, n_features = X.shape
            self.coef = np.linalg.solve(X.T.dot(X) + self.l2_reg * np.identity(n_features), X.T.dot(y))

    def predict(self, X):
        X = self.__handle_features(X)
        return np.dot(self.coef, X.T)
