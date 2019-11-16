import numpy as np


def zip_with_p_value(X):
    return np.array(list(map(lambda x: list(zip(x[0::2], x[1::2])), X)))


def discretize_metric(metric, threshold, neg_threshold=None):
    if neg_threshold is None:
        neg_threshold = -threshold
    return np.array([np.sign(val) if val > threshold or val < neg_threshold else 0 for val in metric])


def get_positive_threshold(metric, critical_p_value):
    return min((metric[np.argwhere((metric[:, 1] > critical_p_value) & (metric[:, 0] > 0)), 0]))


def get_negative_threshold(metric, critical_p_value):
    return max((metric[np.argwhere((metric[:, 1] > critical_p_value) & (metric[:, 0] < 0)), 0]))


def shift_on_zero_value(metric):
    metric = np.copy(metric)
    zero_change = metric[np.argmin(metric[:, 1]), 0]
    metric[:, 0] -= zero_change
    return metric