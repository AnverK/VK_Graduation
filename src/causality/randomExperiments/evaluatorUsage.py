import numpy as np
import utils
from collections import defaultdict

# ====================================================================================================================

# Read data in format of array (n_samples, n_features)

# ====================================================================================================================
from causality.randomExperiments.Generator import Generator

short_metrics_p, long_metrics_p = utils.read_data(dataset_name='feed_top_ab_tests_pool_big_dataset.csv', shift=True)
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]

n_short_metrics = short_metrics.shape[1]
n_long_metrics = long_metrics.shape[1]

removed_short = []
removed_long = [1, 2, 3]

short_metrics = np.delete(short_metrics, removed_short, axis=1)
long_metrics = np.delete(long_metrics, removed_long, axis=1)

# ====================================================================================================================

# Run experiment generator with several distances

# ====================================================================================================================

envGenerator = Generator(short_metrics, long_metrics)
all_dists = defaultdict(lambda: {i: 0 for i in range(short_metrics.shape[1])})

for k, all_stats in enumerate(envGenerator):
    print(k)
    for stat_name, stat in all_stats.items():
        cur_dists = all_dists[stat_name]
        all_dists[stat_name] = {i: cur_dists[i] + stat[i] for i in range(short_metrics.shape[1])}

    if k % 100 == 0:
        for stat_name, stat in all_dists.items():
            print(stat_name, {k: v for k, v in sorted(stat.items(), key=lambda item: item[1])})
    if k == 1000:
        break
