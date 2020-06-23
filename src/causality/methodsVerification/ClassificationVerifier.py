from itertools import combinations

import networkx as nx
import numpy as np
import utils
from causality.methodsVerification.GraphEstimator import GraphEstimator
from causality.pc.CausalGraphBuilder import CausalGraphBuilder
from causality.pc.EdgeOrientation import EdgeOrientation
from causality.pc.independence.GaussIndependenceTest import GaussConditionalIndepTest
from causality.pc.pag.PagConverter import PagConverter
import matplotlib.pyplot as plt


def generate_data(data, structure, n_samples=1000):
    _, n_features = np.shape(data)
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    avg_mu = np.mean(mu)
    avg_std = np.mean(std)
    gen_data = np.random.normal(avg_mu, avg_std, size=(n_samples, n_features))

    noise_level = 0.05
    for v in nx.topological_sort(structure):
        X = list(structure.predecessors(v))
        if len(X) == 0:
            if v < n_features:
                gen_data[:, v] = np.random.normal(mu[v], std[v], n_samples)
            continue
        weights = np.random.uniform(0.2, 0.8, size=len(X))
        neg = np.random.choice(len(X), size=np.random.randint(0, len(X) + 1), replace=False)
        weights[neg] = -weights[neg]
        gen_data[:, v] = np.dot(gen_data[:, X], weights) + np.random.randn(n_samples) * noise_level
    return gen_data[:, :n_features]


def graph_to_dag(graph, perm):
    dag_edges = []
    for x in nx.nodes(graph):
        for v in nx.all_neighbors(graph, x):
            if perm[x] < perm[v]:
                dag_edges.append((x, v))
    return nx.from_edgelist(dag_edges, nx.DiGraph)


def remove_unobservable(graph, observable):
    new_graph = graph.copy()
    for v in nx.nodes(graph):
        if v in observable:
            continue
        for parent in new_graph.predecessors(v):
            for child in new_graph.successors(v):
                new_graph.add_edge(parent, child)
        new_graph.remove_node(v)
    return new_graph


def dag2pag(graph, observable):
    pag = graph.copy()
    for v in nx.nodes(graph):
        if v in observable:
            continue
        for parent in pag.predecessors(v):
            for child in pag.successors(v):
                pag.add_edge(parent, child)
        for a, b in combinations(pag.successors(v), 2):
            pag.add_edge(a, b)
            pag.add_edge(b, a)
        pag.remove_node(v)
    return pag


short_metrics_p, long_metrics_p = utils.read_data(dataset_name='feed_top_ab_tests_pool_big_dataset.csv')
short_metrics = short_metrics_p[:, :, 0]
long_metrics = long_metrics_p[:, :, 0]

n_short_metrics = short_metrics.shape[1]
n_long_metrics = long_metrics.shape[1]

removed_short = []
removed_long = [1, 2, 3]

short_metrics = np.delete(short_metrics, removed_short, axis=1)
long_metrics = np.delete(long_metrics, removed_long, axis=1)

metrics = np.hstack((long_metrics, short_metrics))


def get_prediction(builder, data, orient_threshold):
    builder.fit(data)
    if orient_threshold is None:
        edges = builder.get_edges()
    else:
        indep_test_func = GaussConditionalIndepTest(np.corrcoef(data.T)).gauss_ci_test
        edge_orienter = EdgeOrientation(skeleton=nx.from_edgelist(PagConverter.pag_to_undirected(builder.get_edges())),
                                        sep_sets=builder.get_sepsets(),
                                        n_nodes=data.shape[1],
                                        indep_test_func=indep_test_func,
                                        p_value=orient_threshold)
        edge_orienter.orient_colliders()
        edge_orienter.apply_rules()
        edges = edge_orienter.get_pag()
    if orient_threshold is None:
        return nx.from_edgelist(PagConverter.pag_to_directed(edges), nx.DiGraph)
    else:
        return nx.from_edgelist(PagConverter.pag_to_softly_directed(edges), nx.DiGraph)


f1_skeleton = []
f1_oriented = []

p_value = 1e-2
pc_builder = CausalGraphBuilder(algorithm='pc', p_value=p_value)
fci_builder = CausalGraphBuilder(algorithm='fci+', p_value=p_value)
pc_builder.fit(metrics)
n_nodes = len(metrics.T)
orig_pc_graph = nx.from_edgelist(PagConverter.pag_to_directed(pc_builder.get_edges()), nx.DiGraph)

for _ in range(200):
    true_graph = graph_to_dag(orig_pc_graph, np.random.permutation(n_nodes))
    generated_data = generate_data(metrics, true_graph.copy())
    observable = np.random.choice(n_nodes, size=40, replace=False)
    observed_data = generated_data[:, observable]

    true_graph = dag2pag(true_graph, observable)
    # true_graph = remove_unobservable(true_graph, observable)
    for v in observable:
        if v not in nx.nodes(true_graph):
            true_graph.add_node(v)

    pred_graph = get_prediction(pc_builder, observed_data, 75e-2)
    mapping = {k: v for k, v in enumerate(observable)}
    pred_graph = nx.relabel_nodes(pred_graph, mapping=mapping)

    for v in observable:
        if v not in nx.nodes(pred_graph):
            pred_graph.add_node(v)
    ge = GraphEstimator()
    stat = ge.calc_stat(true_graph, pred_graph)
    f1_skeleton.append(stat['f1'])
    f1_oriented.append(stat['f1\''])

print(np.mean(f1_skeleton), np.std(f1_skeleton), f1_skeleton)
print(np.mean(f1_oriented), np.std(f1_oriented), f1_oriented)

fig1, ax1 = plt.subplots()
labels = ['FCI mod. skeleton', 'FCI mod. oriented']
bplot = ax1.boxplot([f1_skeleton, f1_oriented], labels=labels)
ax1.set_title('Comparing algorithms with Causal Sufficiency')
ax1.set_ylabel('F1')
ax1.set_xlabel('Algorithms')
plt.savefig('fci_modified.png')
