import numpy as np
from causality.pc.pag.PagEdge import PagEdge, ArrowType
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.rinterface as rinterface
import rpy2.robjects as ro
import rpy2.rlike.container as rlc
np.seterr(all='raise')

rpy2.robjects.numpy2ri.activate()
rinterface.initr()
pcalg = importr('pcalg')


class CausalGraphBuilder:

    def __init__(self, algorithm='fci', indep_test='gauss', indep_test_args=None, p_value=0.01, log_path=None,
                 num_cores=1):
        self.sepset = None
        self.edges = None
        self.n_nodes = None
        self.algorithm = algorithm
        self.indep_test = indep_test
        self.indep_test_args = indep_test_args
        self.p_value = p_value
        self.log_path = log_path
        self.num_cores = num_cores
        self._check_arguments()

    def _check_arguments(self):
        if self.num_cores != 1:
            assert self.log_path is None, 'verbose is not able in multi-thread mode'
            assert self.indep_test == 'gauss', 'user defined function couldn\'t be multi-thread'

        if self.indep_test == 'gauss':
            assert self.indep_test_args is None, 'gaussCItest uses non-user-defined arguments'
        else:
            assert callable(self.indep_test), 'indep_test should be "gauss" or callable'

    @staticmethod
    def _from_array_to_edges(array):
        edges = []
        n = len(array)
        for i in range(n):
            for j in range(i + 1, n):
                if array[i][j] == 0:
                    continue
                edges.append(PagEdge(i, j, array[j][i], array[i][j]))
        return edges

    @staticmethod
    def _from_r_graph_to_edges(r_graph):
        g = r_graph.slots['graph']
        r_edge_list = g.slots['edgeL']
        edges = {}
        for i, r_edges in enumerate(r_edge_list):
            if r_edges == rpy2.robjects.NULL:
                return edges.values()
            for j in r_edges[0]:
                if j - 1 < i and (j - 1, i) in edges:
                    edges[j - 1, i] = PagEdge(j - 1, i, ArrowType.ARROW, ArrowType.ARROW)
                else:
                    edges[i, j - 1] = PagEdge(i, j - 1, ArrowType.NONE, ArrowType.ARROW)
        return edges.values()

    def fit(self, X):
        verbose = self.log_path is not None
        if verbose:
            ro.r.sink(self.log_path, type='output')
        if self.indep_test == 'gauss':
            indep_test = pcalg.gaussCItest
            corr = np.corrcoef(X.T)
            nr, nc = corr.shape
            corr = ro.r.matrix(corr, nrow=nr, ncol=nc)
            ro.r.assign("Corr", corr)
            indep_test_args = rlc.TaggedList([corr, len(X)], ['C', 'n'])
        else:
            indep_test = self.indep_test
            if self.indep_test_args is None:
                indep_test_args = ro.NULL
            else:
                indep_test_args = self.indep_test_args
        n_samples, self.n_nodes = X.shape
        if self.algorithm == 'fci':
            res = pcalg.fci(suffStat=indep_test_args, indepTest=indep_test, p=self.n_nodes, alpha=self.p_value,
                            skel_method="stable.fast",
                            numCores=self.num_cores,
                            verbose=verbose)
            g = res.slots['amat']
            self.edges = self._from_array_to_edges(np.array(g, dtype=int))
        elif self.algorithm == 'fci+':
            res = pcalg.fciPlus(suffStat=indep_test_args, indepTest=indep_test, p=self.n_nodes, alpha=self.p_value,
                                verbose=verbose)
            g = res.slots['amat']
            self.edges = self._from_array_to_edges(np.array(g, dtype=int))
        else:
            res = pcalg.pc(suffStat=indep_test_args, indepTest=indep_test, p=self.n_nodes, alpha=self.p_value,
                           skel_method="stable.fast",
                           solve_confl=True,
                           u2pd='relaxed',
                           verbose=verbose,
                           numCores=self.num_cores)
            self.edges = self._from_r_graph_to_edges(res)
        try:
            self.sepset = self.extract_sepsets(res)
        except Exception:
            self.sepset = None

    def get_edges(self):
        assert self.edges is not None, 'Graph should be built by fit() function'
        return self.edges

    def get_sepsets(self):
        assert self.sepset is not None, 'Separation sets are not calculated in fit() function'
        return self.sepset

    def extract_sepsets(self, res):
        if self.algorithm == 'fci+':
            # See my PR here: https://github.com/cran/pcalg/pull/3
            # If you don't have too much variables, please use fci instead.
            # Otherwise, you can try PC algorithm and my orientation algorithm.
            raise Exception("FCI+ doesn't provide separation sets")
        sepset = res.slots['sepset']
        py_sepset = dict()
        for v, other_v in enumerate(list(sepset)):
            for u, sep in enumerate(list(other_v)):
                if sep == rpy2.robjects.NULL:
                    continue
                py_sepset[(min(u, v), max(u, v))] = set([k - 1 for k in list(sep)])
        return py_sepset
