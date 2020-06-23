import networkx as nx


class GraphVerifier:

    def __init__(self):
        self._stat = dict()

    def _calc_tp(self, true_graph, pred_graph):
        return nx.number_of_edges(nx.intersection(true_graph, pred_graph))

    def _calc_fp(self, true_graph, pred_graph):
        return nx.number_of_edges(nx.difference(pred_graph, true_graph))

    def _calc_fn(self, true_graph, pred_graph):
        return nx.number_of_edges(nx.difference(true_graph, pred_graph))

    def _calc_tn(self, true_graph, pred_graph):
        return nx.number_of_edges(nx.intersection(nx.complement(true_graph), nx.complement(pred_graph)))

    def _calc_tp_misorient(self, true_graph, pred_graph, satisfied_cs):
        true_edges = nx.edges(nx.intersection(true_graph.to_undirected(), pred_graph.to_undirected()))
        tp_mis = 0
        for u, v in true_edges:
            if true_graph.has_edge(u, v) and true_graph.has_edge(v, u):
                assert not satisfied_cs, "True graph with CS is DAG"
                if not (pred_graph.has_edge(u, v) and pred_graph.has_edge(v, u)):
                    tp_mis += 1
                continue
            if pred_graph.has_edge(u, v) and pred_graph.has_edge(v, u):
                if satisfied_cs:
                    tp_mis += 0.5
                else:
                    tp_mis += 1
                continue
            if true_graph.has_edge(u, v) and not pred_graph.has_edge(u, v):
                tp_mis += 1
                continue
            if true_graph.has_edge(v, u) and not pred_graph.has_edge(v, u):
                tp_mis += 1
        return tp_mis

    def calc_stat(self, true_graph, pred_graph, satisfied_cs=False):
        self._stat['tp_mis'] = self._calc_tp_misorient(true_graph, pred_graph, satisfied_cs)

        true_graph = true_graph.to_undirected(as_view=True)
        pred_graph = pred_graph.to_undirected(as_view=True)

        self._stat['tp'] = self._calc_tp(true_graph, pred_graph)
        self._stat['fp'] = self._calc_fp(true_graph, pred_graph)
        self._stat['fn'] = self._calc_fn(true_graph, pred_graph)
        self._stat['tn'] = self._calc_tn(true_graph, pred_graph)

        self._stat['tp\''] = self._stat['tp'] - self._stat['tp_mis']
        self._stat['fp\''] = self._stat['fp'] + self._stat['tp_mis']

        if self._stat['tp'] == 0:
            self._stat['prec'] = 0
            self._stat['rec'] = 0
            self._stat['f1'] = 0
        else:
            self._stat['prec'] = self._stat['tp'] / (self._stat['tp'] + self._stat['fp'])
            self._stat['rec'] = self._stat['tp'] / (self._stat['tp'] + self._stat['fn'])
            self._stat['f1'] = 2 * self._stat['prec'] * self._stat['rec'] / (self._stat['prec'] + self._stat['rec'])

        if self._stat['tp\''] == 0:
            self._stat['prec\''] = 0
            self._stat['rec\''] = 0
            self._stat['f1\''] = 0
        else:
            self._stat['prec\''] = self._stat['tp\''] / (self._stat['tp\''] + self._stat['fp\''])
            self._stat['rec\''] = self._stat['tp\''] / (self._stat['tp\''] + self._stat['fn'])
            self._stat['f1\''] = 2 * self._stat['prec\''] * self._stat['rec\''] / (
                    self._stat['prec\''] + self._stat['rec\''])

        return self._stat
