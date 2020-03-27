from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import pdist
from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.pyplot as plt
from fcit import fcit
import networkx as nx
from src import utils
import logging
import pcalg

from src.causality.pc.independence.ConditionalIndependenceTest import kernel_based_conditional_independence
from src.causality.pc.independence.UnconditionalIndependenceTest import kernel_based_indepence

fh = logging.FileHandler('pcalg_005_1(2).log')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                    handlers=[fh])
logger = logging.getLogger(__name__)


def kcit_wrapper(data_matrix, i, j, ks, **kwargs):
    x = data_matrix[:, i].reshape(-1, 1)
    y = data_matrix[:, j].reshape(-1, 1)
    if len(ks) == 0:
        return kernel_based_indepence(x, y)
    else:
        return kernel_based_conditional_independence(x, y, data_matrix[:, list(ks)])


def fcit_wrapper(data_matrix, i, j, ks, **kwargs):
    x = data_matrix[:, i].reshape(-1, 1)
    y = data_matrix[:, j].reshape(-1, 1)
    if len(ks) == 0:
        res = fcit.test(x, y)
    else:
        res = fcit.test(x, y, data_matrix[:, list(ks)])
    return 1.0 if np.isnan(res) else res


n_samples = 100
# print(ttest_1samp(np.repeat(5, n_samples), 5))
# exit(1)
# for i in range(50):
x = np.random.randn(n_samples, 1)
y = x
# Z = np.random.randn(n_samples, 2)
# print(fcit.test(x, y, prop_test=0.2))
print(kernel_based_indepence(x, y, approximate=False))
y = np.random.randn(n_samples, 1)
print(kernel_based_indepence(x, y, approximate=False))
# print(kernel_based_conditional_independence(x, y, Z))
exit(1)

short_metrics_p, long_metrics_p = utils.read_data(shift=True)
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]
metrics = np.hstack((short_metrics, long_metrics))
from itertools import combinations
from time import time

start = time()
values = []
for mx, my, mz in combinations(metrics.reshape((metrics.shape[1], metrics.shape[0], 1)), 3):
    values.append(kernel_based_conditional_independence(mx, my, mz, approximate=True))
print(time() - start)
plt.hist(values, bins='auto')
plt.show()
# print(kernel_based_indepence(metrics[:, 1].reshape(-1, 1), metrics[:, 4].reshape(-1, 1)))
# print(fcit.test(metrics[:, 0].reshape(-1, 1), metrics[:, 2].reshape(-1, 1)))
exit(1)
# print(kernel_based_conditional_independence(metrics[:, 0].reshape(-1, 1), metrics[:, 1].reshape(-1, 1),
#                                             metrics[:, 2].reshape(-1, 1), bs_iters=1e5))
# logger.info('p-value: %s, max_reach: %s' % (5e-2, 1))
G, sep_set = pcalg.estimate_skeleton(fcit_wrapper, metrics, 5e-2, max_reach=1)
nx.write_gpickle(G, 'graph_005_1(2).pckl')
# G = nx.read_gpickle('graph_005_1.pckl')
#
_, n_short_metrics = short_metrics.shape
_, n_long_metrics = long_metrics.shape

node_labels = {}
node_colors = ['g'] * n_short_metrics + ['r'] * n_long_metrics

for i in range(n_short_metrics):
    node_labels[i] = str(i)
for i in range(n_long_metrics):
    node_labels[i + n_short_metrics] = str(i)

edges = nx.edges(G)
nx.draw_networkx(G, pos=nx.circular_layout(G), arrows=False, edgelist=edges, width=5.0,
                 labels=node_labels, node_color=node_colors)
plt.show()
