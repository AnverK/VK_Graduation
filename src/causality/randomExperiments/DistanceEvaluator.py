import networkx as nx


class DistanceEvaluator:
    def __init__(self, graph, target_node=0, inf_dist=100):
        self.graph = graph
        self.target_node = target_node
        self.inf_dist = inf_dist

    def _fetch_dists(self, dists):
        for i in range(nx.number_of_nodes(self.graph)):
            if i != self.target_node and i not in dists.keys():
                dists[i] = self.inf_dist
        dists.pop(self.target_node)
        return dists

    def _directed_distances(self):
        dists = nx.single_source_dijkstra_path_length(nx.reverse(self.graph, copy=False), self.target_node)
        return self._fetch_dists(dists)

    def _undirected_distances(self):
        dists = nx.single_source_dijkstra_path_length(self.graph, self.target_node)
        return self._fetch_dists(dists)

    def get_distances(self):
        if nx.is_directed(self.graph):
            return self._directed_distances()
        else:
            return self._undirected_distances()
