import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate, GridSearchCV

from src.MetricLinearRegression import MetricLinearRegression
from src.utils import *


def neg_mean_error(estimator, X, y_true):
    y_true = discretize_metric(y_true[:, 0], estimator.pos_threshold, estimator.neg_threshold)
    y_pred = estimator.predict(X)
    # print(y_true)
    # print(y_pred)
    return -mean_absolute_error(y_true, y_pred)


def grid_search(X, y):
    lin_reg = MetricLinearRegression()
    grid_reg = np.linspace(0, 10, num=50)
    grid_beta = np.linspace(0, 10, num=50)
    params = {'l2_reg': grid_reg, 'beta': grid_beta}
    gs = GridSearchCV(lin_reg, param_grid=params, scoring=neg_mean_error, iid=False, cv=5, verbose=1, n_jobs=-1)
    gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)


def evaluate_model(X, y, l2_reg, beta, pos_threshold, neg_threshold):
    lin_reg = MetricLinearRegression(l2_reg, beta, pos_threshold=pos_threshold, neg_threshold=neg_threshold)
    scores = cross_validate(lin_reg, X, y, scoring=neg_mean_error, cv=5, n_jobs=-1)
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
# grid_search(X, y)

# short_term = zipped_X[:, 15]
# x = [0, 100]
pos_threshold = get_positive_threshold(y)
neg_threshold = get_negative_threshold(y)
# print(pos_threshold, neg_threshold)
# y1 = [pos_threshold, pos_threshold]
# y2 = [neg_threshold, neg_threshold]
# plt.plot(x, y1, 'b')
# plt.plot(x, y2, 'b')
# plt.plot(y[:, 1], y[:, 0], 'r.')
# plt.show()
# pos_threshold = 0.035
# neg_threshold = -0.035
l2_reg = 0
beta = 0
evaluate_model(X, y, l2_reg, beta, pos_threshold, neg_threshold)
