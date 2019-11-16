import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize


class MetricLinearRegression(BaseEstimator):
    def __init__(self, beta=1, critical_p_value=99, coef=None):
        self.beta = beta
        self.critical_p_value = critical_p_value

    def __objective_function(self, w, X_e, X_u, y_e):
        e_term = sum([np.dot(w, x) * y for x, y in zip(X_e, y_e)])
        u_term = sum([abs(np.dot(w, x)) for x in X_e])
        return -e_term / len(X_e) - self.beta * u_term / len(X_u)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        painted_experiments = np.argwhere(y[:, 1] > self.critical_p_value)
        uncertain_experiments = np.argwhere(y[:, 1] <= self.critical_p_value)
        X_painted = X[painted_experiments].reshape(len(painted_experiments), n_features)
        y_painted = y[painted_experiments, 0].flatten()
        X_uncertain = X[uncertain_experiments].reshape(len(uncertain_experiments), n_features)

        if self.coef is None:
            self.coef = np.zeros(n_features)

        result = minimize(self.__objective_function, self.coef, (X_painted, X_uncertain, y_painted),
                          method='L-BFGS-B')
        self.coef = result.x
        print(self.coef)

    def predict(self, X):
        return np.sign(np.dot(self.coef, X.T))
