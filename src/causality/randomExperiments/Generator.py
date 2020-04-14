import numpy as np
from causality.pc.CausalGraphBuilder import CausalGraphBuilder
from causality.randomExperiments.Evaluator import Evaluator


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

        n_used = np.random.randint(10, n_features + 1)
        used_metrics_ind = np.random.choice(n_features, n_used, replace=False)
        used_metrics = self.X[:, used_metrics_ind]

        noise_level = np.random.uniform(self.min_noise_level, self.max_noise_level)
        noised_metrics = used_metrics + np.random.randn(*used_metrics.shape) * noise_level
        noised_target = self.y + np.random.rand(*self.y.shape) * noise_level
        metrics = np.hstack((noised_target, noised_metrics))
        metrics = metrics[np.random.choice(n_samples, n_samples)]

        model = CausalGraphBuilder(algorithm='fci+', p_value=self.sig_level, num_cores=4)
        model.fit(metrics)
        evaluator = Evaluator(model.get_edges(), n_features + 1, 0)
        dists = {i: 0 for i in range(n_features)}
        eval_dists = evaluator.undirected_distances()
        eval_dists = {k - 1: v for k, v in eval_dists.items()}
        for i, ind in enumerate(used_metrics_ind):
            dists[ind] = eval_dists[i]
        return dists
