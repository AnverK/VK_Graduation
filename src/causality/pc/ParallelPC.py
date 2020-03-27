import math
import os

from rpy2.robjects.packages import importr
import rpy2.rlike.container as rlc
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro
import rpy2.rinterface as rinterface

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


def pcalg_fci(alpha):
    res = pcalg.fci(suffStat=suffStat, indepTest=pcalg.gaussCItest, p=p, alpha=alpha,
                    skel_method="stable.fast",
                    # biCC=True,
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


def draw_pag(pag, title):
    G = pgv.AGraph(directed=True)
    G.graph_attr['label'] = title
    G.graph_attr['labelfontsize'] = 18
    _, n_short_metrics = short_metrics.shape
    _, n_long_metrics = long_metrics.shape
    for i in range(n_long_metrics):
        G.add_node(i)
        G.get_node(i).attr['label'] = str(i)
        G.get_node(i).attr['fillcolor'] = 'red'
        G.get_node(i).attr['style'] = 'filled'
    for i in range(n_short_metrics):
        node_id = i + n_long_metrics
        G.add_node(node_id)
        G.get_node(node_id).attr['label'] = str(i)
        G.get_node(node_id).attr['fillcolor'] = 'green'
        G.get_node(node_id).attr['style'] = 'filled'

    n = len(pag)
    for i in range(n):
        for j in range(i + 1, n):
            if pag[i][j] == 0:
                continue
            G.add_edge(i, j)
            edge = G.get_edge(i, j)
            edge.attr['dir'] = 'both'
            edge.attr['arrowhead'] = orient(pag[i][j])
            edge.attr['arrowtail'] = orient(pag[j][i])
    G.draw(title + '.png', prog='dot')


def pcalg_pc(alpha):
    res = pcalg.pc(suffStat=suffStat, indepTest=fcit_wrapper, p=p, alpha=alpha, skel_method="stable.fast",
                   solve_confl=True,
                   u2pd='relaxed',
                   # numCores=4,
                   NAdelete=True)
    g = res.slots['graph']
    edgeL = g.slots['edgeL']
    pyEdges = {}
    for i, edges in enumerate(edgeL):
        pyEdges[i] = np.array(list(edges[0])) - 1
    return nx.from_dict_of_lists(pyEdges, nx.DiGraph)


def draw_graph(G, title):
    G = nx.nx_agraph.to_agraph(G)
    _, n_short_metrics = short_metrics.shape
    _, n_long_metrics = long_metrics.shape

    G.graph_attr['label'] = title
    G.graph_attr['labelfontsize'] = 18
    for i in range(n_long_metrics):
        G.get_node(i).attr['label'] = str(i)
        G.get_node(i).attr['fillcolor'] = 'red'
        G.get_node(i).attr['style'] = 'filled'
    for i in range(n_short_metrics):
        G.get_node(i + n_long_metrics).attr['label'] = str(i)
        G.get_node(i + n_long_metrics).attr['fillcolor'] = 'green'
        G.get_node(i + n_long_metrics).attr['style'] = 'filled'

    G.draw(title + '.png', prog='dot')


short_metrics_p, long_metrics_p = utils.read_data(dataset_name='feed_top_ab_tests_pool_dataset_new.csv', shift=True)
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]
metrics = np.hstack((long_metrics, short_metrics))
data_matrix = metrics
# metrics = np.array(list(map(lambda metric: utils.apply_l0_regularization(metric, 60), X)))[:, :, 0]
corrolationMatrix = np.corrcoef(metrics.T)
p = metrics.shape[1]
n = metrics.shape[0]

# flatCorr = np.abs(corrolationMatrix).flatten()
# inds = flatCorr.argsort()[::-1]

nr, nc = corrolationMatrix.shape
corr = ro.r.matrix(corrolationMatrix, nrow=nr, ncol=nc)
ro.r.assign("Corr", corr)
suffStat = rlc.TaggedList([corr, n], ['C', 'n'])

# G = pcalg_fci(0.01)
# draw_pag(G, 0.01)

alphas = [1e-2]
pc_title = 'pc_new_p_'
fci_title = 'fci_new_p_'

for alpha in alphas:
    cur_title = fci_title + str(alpha)
    # log_file = open(os.path.join('pcalg_logs', cur_title + '.log'), 'w')
    # sys.stdout = log_file
    # def r_writelog(s):
    #     print('kek')
    #     log_file.write(s)
    # rinterface.set_writeconsole_regular(r_writelog)
    # rinterface.set_writeconsole_warnerror(r_writelog)
    G = pcalg_fci(alpha)
    # draw_pag(G, os.path.join('graphs', cur_title))
    # draw_graph(G, pc_title + str(alpha))
    # print(time() - start)
    # log_file.close()
