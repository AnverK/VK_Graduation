import networkx as nx
from causality.pc.PagEdge import PagEdge, ArrowType


class EdgeOrientation:
    def __init__(self, skeleton, sep_sets, n_nodes=None):
        if n_nodes is None:
            n_nodes = nx.number_of_nodes(skeleton)
        self.n_nodes = n_nodes
        self.skeleton = nx.to_undirected(skeleton)
        self.sep_sets = sep_sets
        self.pag = nx.to_edgelist(self.skeleton)
        self.pag = list(map(lambda edge: PagEdge(edge[0], edge[1], ArrowType.CIRCLE, ArrowType.CIRCLE), self.pag))

    def orient_colliders(self):
        edge_types = dict(map(lambda edge:
                              ((edge[0], edge[1]), ArrowType.CIRCLE),
                              nx.to_edgelist(nx.to_directed(self.skeleton))))
        for x in nx.nodes(self.skeleton):
            for y in nx.neighbors(self.skeleton, x):
                for z in nx.neighbors(self.skeleton, y):
                    if z <= x or z in nx.neighbors(self.skeleton, x):
                        continue
                    assert (x, z) in self.sep_sets
                    if y in self.sep_sets[(x, z)]:
                        continue
                    edge_types[(x, y)] = ArrowType.ARROW
                    edge_types[(z, y)] = ArrowType.ARROW
        for pag_edge in self.pag:
            pag_edge.head_type = edge_types[(pag_edge.v_from, pag_edge.v_to)]
            pag_edge.tail_type = edge_types[(pag_edge.v_to, pag_edge.v_from)]

    def get_pag(self):
        return self.pag
