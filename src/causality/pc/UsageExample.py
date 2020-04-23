import os
import numpy as np

from causality.pc.CausalGraphBuilder import CausalGraphBuilder
from causality.pc.LogParser import LogParser
from causality.pc.pag.PagDrawer import PagDrawer
import utils

# ====================================================================================================================

# Read data in format of array (n_samples, n_features)

# ====================================================================================================================

short_metrics_p, long_metrics_p = utils.read_data(dataset_name='feed_top_ab_tests_pool_dataset_new.csv', shift=True)
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]
metrics = np.hstack((long_metrics, short_metrics))
n_short_metrics = short_metrics.shape[1]
n_long_metrics = long_metrics.shape[1]

# ====================================================================================================================

# Fit this model with CausalGraphBuilder with desired parameters. Be careful with restrictions (they are mentioned
# as CausalGraphBuilder._check_arguments)

# ====================================================================================================================

p_value = 1e-2
raw_log_path = os.path.join('pcalg_logs', 'example.log')
builder = CausalGraphBuilder(algorithm='fci', p_value=p_value, num_cores=1, log_path=raw_log_path)
builder.fit(metrics)

# ====================================================================================================================

# Create labels and colors for features. If either of them is not created, non-informative instance will be provided.
# Create path and title of graph as well.
# Build the graph with instance of builder which was already fitted.

# ====================================================================================================================

graph_title = 'example_graph'
graph_path = os.path.join('graphs', graph_title + '.png')
labels = []
colors = []
for i in range(n_long_metrics):
    labels.append('long_' + str(i))
    colors.append('red')
for i in range(n_short_metrics):
    labels.append('short_' + str(i))
    colors.append('green')
PagDrawer.draw(builder.get_edges(), metrics.shape[1], graph_title, graph_path, labels, colors)

# ====================================================================================================================

# If you saved raw log of builder work, you can fetch it with LogParser class. It also needs labels to provide
# informative output.

# After class was created, one can find information about specific edges (using labels or number of vertex) or
# get the whole fetched log about the graph.

# Please note that currently LogParser supports only FCI logs (not completely)!

# ====================================================================================================================

parser = LogParser(raw_log_path, p_value=p_value, labels=labels)

print(parser.edge_existence(0, 1))
print(parser.edge_existence('long_0', 'long_1'))

# note, edge_direction is order-dependent!
print(parser.edge_direction(4, 0))
print(parser.edge_direction('long_0', 'long_1'))

fetched_log_path = os.path.join('pcalg_logs', 'fetched_example.log')
parser.get_fetched_log(fetched_log_path)
