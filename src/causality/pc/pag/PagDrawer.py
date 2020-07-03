import pygraphviz as pgv
from causality.pc.pag.PagEdge import PagEdge


class PagDrawer:
    def __init__(self):
        pass

    @staticmethod
    def draw(edges, n_nodes, title="", graph_path=None, labels=None, colors=None):
        if labels is None:
            labels = list(map(str, range(n_nodes)))
        if colors is None:
            colors = ['green'] * n_nodes

        G = pgv.AGraph(directed=True)
        G.graph_attr['label'] = title
        G.graph_attr['labelfontsize'] = 18
        for i in range(n_nodes):
            G.add_node(i)
            G.get_node(i).attr['label'] = labels[i]
            G.get_node(i).attr['fillcolor'] = colors[i]
            G.get_node(i).attr['style'] = 'filled'

        for edge_info in edges:
            G.add_edge(edge_info.v_from, edge_info.v_to)
            edge = G.get_edge(edge_info.v_from, edge_info.v_to)
            edge.attr['dir'] = 'both'
            edge.attr['arrowhead'] = PagEdge._orient_pag(edge_info.head_type)
            edge.attr['arrowtail'] = PagEdge._orient_pag(edge_info.tail_type)
        return G.draw(graph_path, prog='dot', format='png')
