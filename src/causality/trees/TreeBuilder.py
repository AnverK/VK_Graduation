from scipy.spatial.distance import pdist, squareform
from src.causality.trees.scores import mutual_info_pairwise
import networkx as nx


class TreeBuilder:

    def __init__(self, score=None):
        if score is None:
            self.score = mutual_info_pairwise
        else:
            self.score = score

    def build_graph(self, data):
        return squareform(pdist(data.T, metric=self.score), force='tomatrix', checks=False)

    def build_tree(self, data):
        graph = self.build_graph(data)
        return nx.maximum_spanning_arborescence(nx.from_numpy_matrix(graph).to_directed())
