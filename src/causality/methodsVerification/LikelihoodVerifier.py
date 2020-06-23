import numpy as np
import utils
import networkx as nx
import matplotlib.pyplot as plt
from causality.pc.CausalGraphBuilder import CausalGraphBuilder
from causality.pc.EdgeOrientation import EdgeOrientation
from causality.pc.independence.GaussIndependenceTest import GaussConditionalIndepTest
from causality.pc.pag.PagConverter import PagConverter
from causality.greedyBuilder.scores import compute_likelihood

# ====================================================================================================================

# Read data in format of array (n_samples, n_features)

# ====================================================================================================================

short_metrics_p, long_metrics_p = utils.read_data(dataset_name='feed_top_ab_tests_pool_big_dataset.csv', shift=True)

short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]

n_short_metrics = short_metrics.shape[1]
n_long_metrics = long_metrics.shape[1]

removed_short = []
removed_long = [1, 2, 3]

short_metrics = np.delete(short_metrics, removed_short, axis=1)
long_metrics = np.delete(long_metrics, removed_long, axis=1)

metrics = np.hstack((long_metrics, short_metrics))
clear_metrics = np.copy(metrics)

# ====================================================================================================================

p_value = 1e-2
pc_builder = CausalGraphBuilder(algorithm='pc', p_value=p_value)

res_pc = []
edges_pc = []
res_pc_modified = []
edges_pc_modified = []

for i in range(100):
    print(i)
    metrics = clear_metrics[:, np.random.permutation(np.shape(clear_metrics)[1])]

    pc_builder.fit(metrics + np.random.randn(*np.shape(clear_metrics)) * 0.01)
    pc_edges = pc_builder.get_edges()
    pc_graph = nx.from_edgelist(PagConverter.pag_to_directed(pc_edges), nx.DiGraph)
    edges_pc.append(nx.number_of_edges(pc_graph))
    likelihood_oriented, _ = compute_likelihood(metrics, pc_graph)
    res_pc.append(sum(likelihood_oriented))

    indep_test_func = GaussConditionalIndepTest(np.corrcoef(metrics.T)).gauss_ci_test
    edge_orienter = EdgeOrientation(skeleton=nx.from_edgelist(PagConverter.pag_to_undirected(pc_edges)),
                                    sep_sets=pc_builder.get_sepsets(),
                                    n_nodes=metrics.shape[1],
                                    indep_test_func=indep_test_func,
                                    p_value=3e-1)
    edge_orienter.orient_colliders()
    edge_orienter.apply_rules(pc=True)
    edges = edge_orienter.get_pag()
    pc_modified_graph = nx.from_edgelist(PagConverter.pag_to_directed(edges), nx.DiGraph)
    edges_pc_modified.append(nx.number_of_edges(pc_modified_graph))
    likelihood_oriented, _ = compute_likelihood(metrics, pc_modified_graph)
    res_pc_modified.append(sum(likelihood_oriented))

print(np.mean(res_pc), np.std(res_pc), res_pc)
print(np.mean(edges_pc), np.std(edges_pc), edges_pc)
print(np.mean(res_pc_modified), np.std(res_pc_modified), res_pc_modified)
print(np.mean(edges_pc_modified), np.std(edges_pc_modified), edges_pc_modified)

fig1, ax1 = plt.subplots()
labels = ['PC', 'PC mod.']
bplot = ax1.boxplot([res_pc, res_pc_modified], labels=labels)
ax1.set_title('Comparing likelihood of algorithms')
ax1.set_ylabel('Likelihood')
ax1.set_xlabel('Algorithms')
plt.savefig('likelihood.png')

fig2, ax2 = plt.subplots()
labels = ['PC', 'PC mod.']
bplot2 = ax2.boxplot([edges_pc, edges_pc_modified], labels=labels)
ax2.set_title('Comparing number of edges of algorithms')
ax2.set_ylabel('Edges')
ax2.set_xlabel('Algorithms')
plt.savefig('likelihood_edges.png')
