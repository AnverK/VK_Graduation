import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, LinearRegression
from src.utils import *


class InstrumentalVariable(BaseEstimator):
    def __init__(self, critical_p_value=95, l2_reg=None):
        self.critical_p_value = critical_p_value
        self.coef_ = None
        self.intercept_ = None
        self.l2_reg = l2_reg

    def __handle_features(self, X):
        return np.array(list(map(lambda metric: apply_l0_regularization(metric, self.critical_p_value), X)))[:, :, 0]

    def fit(self, X, y):
        X = self.__handle_features(X)
        if self.l2_reg is None:
            model = LinearRegression()
        else:
            model = Ridge(self.l2_reg, solver='lsqr')
        model.fit(X, y)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

    def predict(self, X):
        X = self.__handle_features(X)
        return np.dot(self.coef_, X.T) + self.intercept_
