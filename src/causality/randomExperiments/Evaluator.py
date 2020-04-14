from causality.pc.PagEdge import PagEdge, ArrowType
import networkx as nx


class Evaluator:
    def __init__(self, pag_edges, n_nodes, target_node=0, inf_dist=100):
        self.n_nodes = n_nodes
        self.pag_edges = pag_edges
        self.target_node = target_node
        self.inf_dist = inf_dist

        self.undir_graph = nx.Graph()
        self.undir_graph.add_nodes_from(range(n_nodes))
        self.undir_graph.add_edges_from(self.pag_to_undirected(pag_edges))

        self.dir_graph = nx.DiGraph()
        self.dir_graph.add_nodes_from(range(n_nodes))
        self.dir_graph.add_edges_from(self.pag_to_directed(pag_edges))

        self.semidir_graph = nx.DiGraph()
        self.semidir_graph.add_nodes_from(range(n_nodes))
        self.semidir_graph.add_edges_from(self.pag_to_semidirected(pag_edges))

    @staticmethod
    def pag_to_directed(pag_edges):
        dir_edges = []
        for edge in pag_edges:
            assert isinstance(edge, PagEdge), "Graph should be presented as list of PAG edges"
            if edge.tail_type == ArrowType.NONE and edge.head_type == ArrowType.ARROW:
                dir_edges.append((edge.v_from, edge.v_to))
                continue
            if edge.tail_type == ArrowType.ARROW and edge.head_type == ArrowType.NONE:
                dir_edges.append((edge.v_to, edge.v_from))
        return dir_edges

    @staticmethod
    def pag_to_undirected(pag_edges):
        undir_edges = []
        for edge in pag_edges:
            assert isinstance(edge, PagEdge), "Graph should be presented as list of PAG edges"
            undir_edges.append((edge.v_from, edge.v_to))
        return undir_edges

    @staticmethod
    def pag_to_semidirected(pag_edges):
        edges = []
        for edge in pag_edges:
            assert isinstance(edge, PagEdge), "Graph should be presented as list of PAG edges"
            if not (edge.head_type == ArrowType.ARROW and edge.tail_type == ArrowType.NONE):
                edges.append((edge.v_to, edge.v_from))
            if not (edge.head_type == ArrowType.NONE and edge.tail_type == ArrowType.ARROW):
                edges.append((edge.v_from, edge.v_to))
        return edges

    def _fetch_dists(self, dists):
        if len(dists) == self.n_nodes:
            dists.pop(self.target_node)
            return dists
        for i in range(self.n_nodes):
            if i != self.target_node and i not in dists.keys():
                dists[i] = self.inf_dist
        dists.pop(self.target_node)
        return dists

    def semidirected_distances(self):
        dists = nx.single_source_dijkstra_path_length(nx.reverse(self.semidir_graph, copy=False), self.target_node)
        return self._fetch_dists(dists)

    def undirected_distances(self):
        dists = nx.single_source_dijkstra_path_length(self.undir_graph, self.target_node)
        return self._fetch_dists(dists)

    def directed_distances(self):
        dists = nx.single_source_dijkstra_path_length(nx.reverse(self.dir_graph, copy=False), self.target_node)
        return self._fetch_dists(dists)
