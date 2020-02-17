from src.causality.trees.TreeBuilder import TreeBuilder
from src.causality.trees.GraphBuilder import GraphBuilder
import src.utils as utils
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

short_metrics_p, long_metrics_p = utils.read_data(shift=True)
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]

target_metric_p = long_metrics_p[:, 0, :]  # <--- here you can choose target (0, 1, 2, 3)
target_metric = target_metric_p[:, 0]

metrics = np.hstack((short_metrics, long_metrics))
builder = GraphBuilder()
G = builder.build_graph(metrics)
nx.write_gpickle(G, 'graph.pckl')
# G = nx.read_gpickle('graph.pckl')

# G = nx.from_scipy_sparse_matrix(tree)
# from time import perf_counter
# start = perf_counter()
# for i in range(10):
#     tree = builder.build_tree(metrics)
#     G = nx.from_scipy_sparse_matrix(tree)
# end = perf_counter()
# print(end - start)

_, n_short_metrics = short_metrics.shape
_, n_long_metrics = long_metrics.shape

node_labels = {}
node_colors = ['g'] * n_short_metrics + ['r'] * n_long_metrics

for i in range(n_short_metrics):
    node_labels[i] = str(i)
for i in range(n_long_metrics):
    node_labels[i + n_short_metrics] = str(i)

edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw_networkx(G, pos=nx.circular_layout(G), arrows=True, edgelist=edges, edge_color=weights, width=5.0,
                 edge_vmin=0, edge_cmap=plt.get_cmap("Blues"), labels=node_labels, node_color=node_colors)
plt.show()
