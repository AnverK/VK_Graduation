import numpy as np


def zip_with_p_value(X):
    return np.array(list(map(lambda x: list(zip(x[0::2], x[1::2])), X)))


def discretisize_metric(metric, threshold):
    return np.array([np.sign(val) if abs(val) > threshold else 0 for val in metric])

def shift_on_zero_value(metric):
    metric = np.copy(metric)
    zero_change = metric[np.argmin(metric[:, 1]), 0]
    metric[:, 0] -= zero_change
    return metric

def find_index_of_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()