import numpy as np
import utils

# ====================================================================================================================

# Read data in format of array (n_samples, n_features)

# ====================================================================================================================
from causality.randomExperiments.Generator import Generator

short_metrics_p, long_metrics_p = utils.read_data(dataset_name='feed_top_ab_tests_pool_dataset_new.csv', shift=True)
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]

n_short_metrics = short_metrics.shape[1]
n_long_metrics = long_metrics.shape[1]

removed_short = []
removed_long = [1, 2, 3]

short_metrics = np.delete(short_metrics, removed_short, axis=1)
long_metrics = np.delete(long_metrics, removed_long, axis=1)

# ====================================================================================================================

# Run experiment generator. Currently each value from generator returns distances in undirected graph to each short
# metric.

# ====================================================================================================================

envGenerator = Generator(short_metrics, long_metrics)
sum_m = {i: 0 for i in range(short_metrics.shape[1])}
print(sum_m)
for k, m in enumerate(envGenerator):
    print(k)
    if k == 100:
        break
    sum_m = {i: sum_m[i] + m[i] for i in range(short_metrics.shape[1])}
    print(sum_m)
