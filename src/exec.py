import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error
from src.utils import zip_with_p_value, shift_on_zero_value, discretisize_metric, find_index_of_nearest
from src.MetricLinearRegression import MetricLinearRegression
import matplotlib.pyplot as plt


def painted_accuracy(estimator, X, y_true):
    # n_samples, n_features = X.shape
    # painted_inds = np.argwhere(y_true[:, 1] > estimator.critical_p_value)
    # y_true = y_true[painted_inds, 0].flatten()
    # X = X[painted_inds].reshape(len(painted_inds), n_features)
    y_true = discretisize_metric(y_true[:, 0], estimator.threshold)
    y_pred = estimator.predict(X)
    print(y_true)
    print(y_pred)
    return -mean_absolute_error(y_true, y_pred)


def grid_search(X, y):
    lin_reg = MetricLinearRegression(threshold=0.035)
    grid_reg = np.linspace(0, 10, num=50)
    grid_beta = np.linspace(0, 10, num=50)
    # grid_threshold = np.linspace(0, 0.1, num=10)
    params = {'l2_reg': grid_reg, 'beta': grid_beta}
    gs = GridSearchCV(lin_reg, param_grid=params, scoring=painted_accuracy, iid=False, cv=5, verbose=1, n_jobs=-1)
    gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)


def evaluate_model(X, y, l2_reg, beta, threshold):
    lin_reg = MetricLinearRegression(l2_reg, beta, threshold=threshold)
    scores = cross_validate(lin_reg, X, y, scoring=painted_accuracy, cv=20, n_jobs=-1)
    print(np.mean(scores['test_score']))


dataset_path = './dataset/feed_top_ab_tests_pool_dataset.csv'

df = pd.read_csv(dataset_path)
data = df.to_numpy()
LONG_TERM_COUNT = 4
y = data[:, :LONG_TERM_COUNT * 2]
X = data[:, LONG_TERM_COUNT * 2:-1]
zipped_X = zip_with_p_value(X)
zipped_y = zip_with_p_value(y)

zipped_X[:] = [shift_on_zero_value(short_term_metric) for short_term_metric in zipped_X[:]]

X = zipped_X[:, :, 0]
y = zipped_y[:, 2, :]

y = shift_on_zero_value(y)
# threshold = max(np.abs(y[np.argwhere(y[:, 1] < 99), 0]))
# print(threshold)
# threshold = 0.1
# print(y)
# grid_search(X, y)

# short_term = zipped_X[:, 15]
# x = [0, 100]
# threshold = 0.035
# y1 = [threshold, threshold]
# y2 = [-threshold, -threshold]
# plt.plot(x, y1, 'b')
# plt.plot(x, y2, 'b')
# plt.plot(y[:, 1], y[:, 0], 'r.')
# plt.show()
l2_reg = 3.8775510204081636
beta = 0
threshold = 0.035
evaluate_model(X, y, l2_reg, beta, threshold)
