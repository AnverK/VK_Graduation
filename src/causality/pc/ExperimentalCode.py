import math
import os

from rpy2.robjects.packages import importr
import rpy2.rlike.container as rlc
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro
import rpy2.rinterface as rinterface
from sklearn.preprocessing import normalize

rpy2.robjects.numpy2ri.activate()
rinterface.initr()
pcalg = importr('pcalg')

from scipy.stats import norm

import utils

from causality.pc.independence.ConditionalIndependenceTest import kernel_based_conditional_independence
from causality.pc.independence.UnconditionalIndependenceTest import kernel_based_indepence
import numpy as np
import networkx as nx
import pygraphviz as pgv
import graphviz as gv
from fcit import fcit
from time import time
import sys


def fcit_wrapper(i, j, ks, ignored):
    start = time()
    print(int(i[0] - 1), int(j[0] - 1), len(ks), )
    x = data_matrix[:, int(i[0] - 1)].reshape(-1, 1)
    y = data_matrix[:, int(j[0] - 1)].reshape(-1, 1)
    if len(ks) == 0:
        res = fcit.test(x, y)
    else:
        res = fcit.test(x, y, data_matrix[:, np.array(list(map(int, ks))) - 1])
    print(time() - start)
    return res


def kcit_wrapper(i, j, ks, ignored):
    start = time()
    print(int(i[0] - 1), int(j[0] - 1), len(ks), )
    x = data_matrix[:, int(i[0] - 1)].reshape(-1, 1)
    y = data_matrix[:, int(j[0] - 1)].reshape(-1, 1)
    if len(ks) == 0:
        res = kernel_based_indepence(x, y, approximate=False)
    else:
        res = kernel_based_conditional_independence(x, y, data_matrix[:, np.array(list(map(int, ks))) - 1])
    print(time() - start)
    return res


global test_nums
test_nums = 0


def gauss_wrapper(i, j, ks, ignored):
    global test_nums
    test_nums += 1
    print(test_nums)
    C = corrolationMatrix
    i = int(i[0] - 1)
    j = int(j[0] - 1)
    cut_at = 0.9999999
    if len(ks) == 0:
        r = C[i, j]
    elif len(ks) == 1:
        k = int(ks[0] - 1)
        r = (C[i, j] - C[i, k] * C[j, k]) / math.sqrt((1 - C[j, k] ** 2) * (1 - C[i, k] ** 2))
    else:
        ks = list(map(lambda x: int(x) - 1, ks))
        m = C[np.ix_([i] + [j] + ks, [i] + [j] + ks)]
        PM = np.linalg.pinv(m)
        r = -PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))

    r = min(cut_at, max(-1 * cut_at, r))
    res = math.sqrt(n - len(ks) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    return 2 * (1 - norm.cdf(abs(res)))


def gauss_ci_test(i, j, ks, ignored):
    C = correlation_matrix
    cut_at = 0.9999999
    if len(ks) == 0:
        r = C[i, j]
    elif len(ks) == 1:
        k = ks[0]
        print(C[i, j], C[i, k], C[j, k])
        r = (C[i, j] - C[i, k] * C[j, k]) / math.sqrt((1 - C[j, k] ** 2) * (1 - C[i, k] ** 2))
    else:
        m = C[np.ix_([i] + [j] + ks, [i] + [j] + ks)]
        PM = np.linalg.pinv(m)
        r = -PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    r = min(cut_at, max(-1 * cut_at, r))
    res = math.sqrt(n - len(ks) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    print(r, 2 * (1 - norm.cdf(abs(res))))
    return 2 * (1 - norm.cdf(abs(res)))


def pcalg_fci(alpha):
    res = pcalg.fci(suffStat=suffStat, indepTest=pcalg.gaussCItest, p=p, alpha=alpha,
                    skel_method="stable.fast",
                    # numCores=4,
                    verbose=True,
                    NAdelete=True)
    g = res.slots['amat']
    return np.array(g, dtype=int)


def pcalg_fciPlus(alpha):
    res = pcalg.fciPlus(suffStat=suffStat, indepTest=gauss_wrapper, p=p, alpha=alpha)
    g = res.slots['amat']
    return np.array(g, dtype=int)


def orient(type):
    if type == 1:
        return 'odot'
    elif type == 2:
        return 'normal'
    else:
        assert type == 3
        return 'none'


def draw_pag(pag, title, labels, colors):
    G = pgv.AGraph(directed=True)
    G.graph_attr['label'] = title
    G.graph_attr['labelfontsize'] = 18
    n = len(pag)
    assert n == len(labels)
    assert n == len(colors)
    for i in range(n):
        G.add_node(i)
        G.get_node(i).attr['label'] = labels[i]
        G.get_node(i).attr['fillcolor'] = colors[i]
        G.get_node(i).attr['style'] = 'filled'
    for i in range(n):
        for j in range(i + 1, n):
            if pag[i][j] == 0:
                continue
            G.add_edge(i, j)
            edge = G.get_edge(i, j)
            edge.attr['dir'] = 'both'
            edge.attr['arrowhead'] = orient(pag[i][j])
            edge.attr['arrowtail'] = orient(pag[j][i])
    G.draw(os.path.join('graphs', title + '.png'), prog='dot')


def pcalg_pc(alpha):
    # labels = []
    # for i in range(4):
    #     labels.append('long_' + str(i))
    # for i in range(18):
    #     labels.append('short_' + str(i - 4))
    # labels = ro.sequence_to_vector(labels)
    res = pcalg.pc(suffStat=suffStat, indepTest=pcalg.gaussCItest,
                   alpha=alpha,
                   skel_method="stable.fast",
                   p=p,
                   # labels=labels,
                   solve_confl=True,
                   u2pd='relaxed',
                   # numCores=4,
                   verbose=True,
                   NAdelete=True)
    g = res.slots['graph']
    edgeL = g.slots['edgeL']
    pyEdges = {}
    for i, edges in enumerate(edgeL):
        pyEdges[i] = np.array(list(edges[0])) - 1
    return nx.from_dict_of_lists(pyEdges, nx.DiGraph)


def draw_graph(G, title, labels, colors):
    G = nx.nx_agraph.to_agraph(G)
    n = len(G)
    assert n == len(labels)
    assert n == len(colors)
    G.graph_attr['label'] = title
    G.graph_attr['labelfontsize'] = 18
    for i in range(n):
        G.add_node(i)
        G.get_node(i).attr['label'] = labels[i]
        G.get_node(i).attr['fillcolor'] = colors[i]
        G.get_node(i).attr['style'] = 'filled'

    G.draw(title + '.png', prog='dot')


def suffstat_with_noise(metrics, noise_level=0.01):
    noise_metrics = np.hstack((long_metrics, short_metrics)) + np.random.randn(*metrics.shape) * noise_level
    correlation_matrix = np.corrcoef(noise_metrics.T)
    n = len(metrics)
    nr, nc = correlation_matrix.shape
    corr = ro.r.matrix(correlation_matrix, nrow=nr, ncol=nc)
    ro.r.assign("Corr", corr)
    return rlc.TaggedList([corr, n], ['C', 'n'])


# x = np.random.randn(6000, 1)
# y = np.square(x)
# n_samples = int(1e6)
# r = np.random.randn(n_samples, 1)
# rx = np.random.randn(n_samples, 1)
# ry = np.random.randn(n_samples, 1)
# x = (r + rx) + np.random.randn(n_samples, 1) / 100
# y = (r + ry) + np.random.randn(n_samples, 1) / 100
# short_metrics = np.hstack((r, rx, ry, x, y))
# long_metrics = x + y + np.random.randn(n_samples, 1) / 100
# long_metrics = np.random.randn(6000, 1)
# long_metrics = np.array(x + y).reshape(-1, 1)
short_metrics_p, long_metrics_p = utils.read_data(dataset_name='feed_top_ab_tests_pool_dataset_new.csv', shift=True)
short_metrics = short_metrics_p[:, :, 0]
short_metrics = np.delete(short_metrics, obj=10, axis=1)
# short_metrics = np.hstack((short_metrics, short_metrics[:, 0].reshape(-1, 1)))
long_metrics = long_metrics_p[:, :, 0]
long_metrics = np.delete(long_metrics, obj=2, axis=1)
metrics = np.hstack((long_metrics, short_metrics))

p = metrics.shape[1]
# n = metrics.shape[0]
# suff_stat = suffstat_with_noise(metrics, 0.001)
# correlation_matrix = np.array(suff_stat[0])
# gauss_ci_test(4, 7, [22], None)
# exit(0)
# alphas = [1e-2]
# noises = [0]
suffStat = suffstat_with_noise(metrics, noise_level=0)
G = pcalg_pc(alpha=1e-2)
labels = []
colors = []
for i in range(4):
    if i != 2:
        labels.append('long_' + str(i))
        colors.append('red')
for i in range(18):
    if i != 10:
        labels.append('short_' + str(i))
        colors.append('green')
fci_title = 'aapc_new_p_0.01_noise_0'
draw_graph(G, fci_title, labels, colors)
# noises = [0, 1e-3, 2e-3, 5e-3, 7e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]
# pc_title = 'pc_corr_experiment_new_p_'

# pc_title = 'fci_noise_simple_test'
# for noise in noises:
#     # cur_title = fci_title + str(noise)
#     # log_file = open(os.path.join('pcalg_logs', cur_title + '.log'), 'w')
#     # sys.stdout = log_file
#     # def r_writelog(s):
#     #     print('kek')
#     #     log_file.write(s)
#     # rinterface.set_writeconsole_regular(r_writelog)
#     # rinterface.set_writeconsole_warnerror(r_writelog)
#     start = time()
#     suffStat = suffstat_with_noise(metrics, noise)
#     # G = pcalg_pc(alpha)
#     G = pcalg_fci(alphas[0])
#     draw_pag(G, fci_title)
#     # draw_graph(G, pc_title + str(alpha))
#     print(time() - start)
#     # log_file.close()
