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


short_metrics_p, long_metrics_p = utils.read_data(dataset_name='feed_top_ab_tests_pool_big_dataset.csv', shift=True)
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]

metrics = np.hstack((long_metrics, short_metrics))
builder = GraphBuilder()
G = builder.build_graph(metrics)
draw_graph(G)
