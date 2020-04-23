from causality.pc.pag.PagEdge import ArrowType
import networkx as nx


class PagConverter:
    def __init__(self, pag_edges, n_nodes):
        self.pag_edges = pag_edges
        self.n_nodes = n_nodes

        # skeleton of the graph
        self.undirected = None

        # directed is a graph with all edges directed as in pag, but "o" transforms to ">"
        self.directed = None

        # strictly_directed is a graph with edges A *-> B transforms in eliminating edge B -> A
        self.strictly_directed = None

        # softly_directed is a graph with edges A -> B transforms in eliminating edge B -> A
        self.softly_directed = None

    @staticmethod
    def pag_to_directed(pag_edges):
        edges = []
        for edge in pag_edges:
            if edge.head_type != ArrowType.NONE:
                edges.append((edge.v_from, edge.v_to))
            if edge.tail_type != ArrowType.NONE:
                edges.append((edge.v_to, edge.v_from))
        return edges

    @staticmethod
    def pag_to_undirected(pag_edges):
        edges = []
        for edge in pag_edges:
            edges.append((edge.v_from, edge.v_to))
        return edges

    @staticmethod
    def pag_to_strictly_directed(pag_edges):
        edges = []
        for edge in pag_edges:
            if edge.head_type != ArrowType.ARROW:
                edges.append((edge.v_to, edge.v_from))
            if edge.tail_type != ArrowType.ARROW:
                edges.append((edge.v_from, edge.v_to))
        return edges

    @staticmethod
    def pag_to_softly_directed(pag_edges):
        edges = []
        for edge in pag_edges:
            if not (edge.tail_type == ArrowType.NONE and edge.head_type == ArrowType.ARROW):
                edges.append((edge.v_from, edge.v_to))
            if not (edge.tail_type == ArrowType.ARROW and edge.head_type == ArrowType.NONE):
                edges.append((edge.v_to, edge.v_from))
        return edges

    def get_pag(self):
        return self.pag_edges

    def get_undirected(self):
        if self.undirected is None:
            self.undirected = nx.Graph()
            self.undirected.add_nodes_from(range(self.n_nodes))
            self.undirected.add_edges_from(self.pag_to_undirected(self.pag_edges))
        return self.undirected

    def get_directed(self):
        if self.directed is None:
            self.directed = nx.DiGraph()
            self.directed.add_nodes_from(range(self.n_nodes))
            self.directed.add_edges_from(self.pag_to_directed(self.pag_edges))
        return self.directed

    def get_strictly_directed(self):
        if self.strictly_directed is None:
            self.strictly_directed = nx.DiGraph()
            self.strictly_directed.add_nodes_from(range(self.n_nodes))
            self.strictly_directed.add_edges_from(self.pag_to_strictly_directed(self.pag_edges))
        return self.strictly_directed

    def get_softly_directed(self):
        if self.softly_directed is None:
            self.softly_directed = nx.DiGraph()
            self.softly_directed.add_nodes_from(range(self.n_nodes))
            self.softly_directed.add_edges_from(self.pag_to_softly_directed(self.pag_edges))
        return self.softly_directed
