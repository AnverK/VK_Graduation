from causality.greedyBuilder.TreeBuilder import TreeBuilder
from causality.greedyBuilder.ClimbingHills import GraphBuilder
import utils as utils
import numpy as np
import networkx as nx


def draw_graph(G):
    G = nx.nx_agraph.to_agraph(G)
    _, n_short_metrics = short_metrics.shape
    _, n_long_metrics = long_metrics.shape

    G.graph_attr['labelfontsize'] = 18
    for i in range(n_long_metrics):
        G.get_node(i).attr['label'] = str(i)
        G.get_node(i).attr['fillcolor'] = 'red'
        G.get_node(i).attr['style'] = 'filled'
    for i in range(n_short_metrics):
        G.get_node(i + n_long_metrics).attr['label'] = str(i)
        G.get_node(i + n_long_metrics).attr['fillcolor'] = 'green'
        G.get_node(i + n_long_metrics).attr['style'] = 'filled'

    G.draw('suboptimal_graph.png', prog='dot')


short_metrics_p, long_metrics_p = utils.read_data(shift=True)
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]

metrics = np.hstack((long_metrics, short_metrics))
builder = GraphBuilder()
G = builder.build_graph(metrics)
draw_graph(G)
# nx.write_gpickle(G, 'graph.pckl')
# G = nx.read_gpickle('graph.pckl')

# G = nx.from_scipy_sparse_matrix(tree)
# from time import perf_counter
# start = perf_counter()
# for i in range(10):
#     tree = builder.build_tree(metrics)
#     G = nx.from_scipy_sparse_matrix(tree)
# end = perf_counter()
# print(end - start)

# _, n_short_metrics = short_metrics.shape
# _, n_long_metrics = long_metrics.shape
#
# node_labels = {}
# node_colors = ['g'] * n_short_metrics + ['r'] * n_long_metrics
#
# for i in range(n_short_metrics):
#     node_labels[i] = str(i)
# for i in range(n_long_metrics):
#     node_labels[i + n_short_metrics] = str(i)
#
# edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
# nx.draw_networkx(G, pos=nx.circular_layout(G), arrows=True, edgelist=edges, edge_color=weights, width=5.0,
#                  edge_vmin=0, edge_cmap=plt.get_cmap("Blues"), labels=node_labels, node_color=node_colors)
# plt.show()
