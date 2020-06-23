import numpy as np
from causality.pc.CausalGraphBuilder import CausalGraphBuilder
from causality.pc.pag.PagConverter import PagConverter
from causality.randomExperiments.DistanceEvaluator import DistanceEvaluator


class Generator:
    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.sig_level = 1e-2
        self.min_noise_level = 0
        self.max_noise_level = 0.01

    def __iter__(self):
        return self

    def __next__(self):
        n_samples, n_features = self.X.shape

        n_used = np.random.randint(8, n_features + 1)
        used_metrics_ind = np.random.choice(n_features, n_used, replace=False)
        used_metrics = self.X[:, used_metrics_ind]

        noise_level = np.random.uniform(self.min_noise_level, self.max_noise_level)
        noised_metrics = used_metrics + np.random.randn(*used_metrics.shape) * noise_level
        noised_target = self.y + np.random.rand(*self.y.shape) * noise_level
        metrics = np.hstack((noised_target, noised_metrics))
        metrics = metrics[np.random.choice(n_samples, n_samples)]

        model = CausalGraphBuilder(algorithm='fci+', p_value=self.sig_level)
        model.fit(metrics)

        converter = PagConverter(model.get_edges(), n_used + 1)
        stat_names = ['undirected_distances',
                      'softly_directed_distances',
                      'strictly_directed_distances']

        graphs = [converter.get_undirected(),
                  converter.get_softly_directed(),
                  converter.get_strictly_directed()]

        inf_dist = 1e6
        all_dists = {}
        for stat_ind, stat_name in enumerate(stat_names):
            evaluator = DistanceEvaluator(graphs[stat_ind], 0, inf_dist=inf_dist)
            graph_dists = {i: 0 for i in range(n_features)}
            eval_dists = evaluator.get_distances()
            for i, ind in enumerate(used_metrics_ind):
                graph_dists[ind] = eval_dists[i + 1]  # because target node is 0, so we actually start from 1
            all_dists[stat_name] = graph_dists
        return all_dists
