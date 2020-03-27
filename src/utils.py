import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def zip_with_p_value(X):
    return np.array(list(map(lambda x: list(zip(x[0::2], x[1::2])), X)))


def discretize_metric(metric, threshold, neg_threshold=None):
    if neg_threshold is None:
        neg_threshold = -threshold
    return np.array([np.sign(val) if val > threshold or val < neg_threshold else 0 for val in metric])


def get_positive_threshold(metric, critical_p_value=99):
    return min((metric[np.argwhere((metric[:, 1] > critical_p_value) & (metric[:, 0] > 0)), 0]))


def get_negative_threshold(metric, critical_p_value=99):
    return max((metric[np.argwhere((metric[:, 1] > critical_p_value) & (metric[:, 0] < 0)), 0]))


def shift_on_zero_value(metric):
    metric = np.copy(metric)
    zero_change = metric[np.argmin(metric[:, 1]), 0]
    metric[:, 0] -= zero_change
    return metric


def extract_painted_inds(metric, threshold):
    return np.argwhere(metric[:, 1] > threshold).flatten()


def apply_l0_regularization(metric, threshold=99):
    metric = np.copy(metric)
    zero_inds = np.argwhere(metric[:, 1] < threshold).flatten()
    metric[zero_inds, 0] = 0
    return metric


def get_outlier_experiments(X, nu=0.1, gamma='scale'):
    svm = OneClassSVM(gamma=gamma, nu=nu)
    pred = svm.fit_predict(X, None)
    return np.argwhere(pred == -1).flatten()

def read_metrics(df, short=True):
    if short:
        pattern = 'short_term_'
    else:
        pattern = 'long_term_'
    metrics_p = []
    n_metrics = len(list(filter(lambda s: s.startswith(pattern), df.columns))) // 2
    for i in range(n_metrics):
        value = pattern + str(i)
        p_value = value + '_p_value'
        metrics_p.append(list(zip(df.get(value), df.get(p_value))))
    return np.array(metrics_p).swapaxes(0, 1)


def read_data(dataset_name, shift=True):
    dataset_path = os.path.join(ROOT_DIR, 'dataset', dataset_name)

    df = pd.read_csv(dataset_path)
    l_metrics_p = read_metrics(df, short=False)
    s_metrics_p = read_metrics(df, short=True)

    if shift:
        s_metrics_p = np.array(
            [shift_on_zero_value(short_metric) for short_metric in s_metrics_p.swapaxes(0, 1)]).swapaxes(0, 1)
        l_metrics_p = np.array(
            [shift_on_zero_value(long_metric) for long_metric in l_metrics_p.swapaxes(0, 1)]).swapaxes(0, 1)

    return s_metrics_p, l_metrics_p
