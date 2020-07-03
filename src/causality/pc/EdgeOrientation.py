import networkx as nx
from causality.pc.pag.PagEdge import PagEdge, ArrowType


class EdgeOrientation:
    def __init__(self, skeleton, sep_sets, indep_test_func=None, threshold=0, n_nodes=None):
        if n_nodes is None:
            n_nodes = nx.number_of_nodes(skeleton)
        self.n_nodes = n_nodes
        self.skeleton = nx.to_undirected(skeleton)
        self.sep_sets = sep_sets
        self.pag = nx.to_edgelist(self.skeleton)
        self.pag = list(map(lambda edge: PagEdge(edge[0], edge[1], ArrowType.CIRCLE, ArrowType.CIRCLE), self.pag))
        self.indep_test_func = indep_test_func
        self.threshold = threshold

    def orient_colliders(self, safe=True):
        edge_types = dict(map(lambda edge:
                              ((edge[0], edge[1]), ArrowType.CIRCLE),
                              nx.to_edgelist(nx.to_directed(self.skeleton))))
        for x in nx.nodes(self.skeleton):
            for z in nx.non_neighbors(self.skeleton, x):
                if z <= x:
                    continue
                for y in nx.common_neighbors(self.skeleton, x, z):
                    assert (x, z) in self.sep_sets
                    sep_set = self.sep_sets[(x, z)]
                    if y in sep_set:
                        continue
                    print(x, z, y, list(sep_set))
                    print(self.indep_test_func(x, z, list(sep_set)))
                    print(self.indep_test_func(x, z, list(sep_set | {y})))
                    if safe and self.indep_test_func(x, z, list(sep_set)) - \
                            self.indep_test_func(x, z, list(sep_set | {y})) > self.threshold:
                        edge_types[(x, y)] = ArrowType.ARROW
                        edge_types[(z, y)] = ArrowType.ARROW
        for pag_edge in self.pag:
            pag_edge.head_type = edge_types[(pag_edge.v_from, pag_edge.v_to)]
            pag_edge.tail_type = edge_types[(pag_edge.v_to, pag_edge.v_from)]

    def rule_1(self, edge_types):
        any_changed = False
        for x in nx.nodes(self.skeleton):
            for z in nx.non_neighbors(self.skeleton, x):
                for y in nx.common_neighbors(self.skeleton, x, z):
                    if edge_types[(x, y)] != ArrowType.ARROW:
                        continue
                    if edge_types[(z, y)] != ArrowType.CIRCLE:
                        continue
                    edge_types[(z, y)] = ArrowType.NONE
                    edge_types[(y, z)] = ArrowType.ARROW
                    any_changed = True
        return any_changed

    def rule_2(self, edge_types):
        any_changed = False
        for x in nx.nodes(self.skeleton):
            for z in nx.neighbors(self.skeleton, x):
                if edge_types[(x, z)] != ArrowType.CIRCLE:
                    continue
                for y in nx.common_neighbors(self.skeleton, x, z):
                    if edge_types[(x, y)] == ArrowType.ARROW and edge_types[(y, x)] == ArrowType.NONE \
                            and edge_types[(y, z)] == ArrowType.ARROW:
                        edge_types[(x, z)] = ArrowType.ARROW
                        any_changed = True
        return any_changed

    def rule_3(self, edge_types):
        any_changed = False
        for x in nx.nodes(self.skeleton):
            for z in nx.non_neighbors(self.skeleton, x):
                for a in nx.common_neighbors(self.skeleton, x, z):
                    if edge_types[(x, a)] != ArrowType.ARROW and edge_types[(z, a)] != ArrowType.ARROW:
                        continue
                    for b in nx.common_neighbors(self.skeleton, x, z):
                        if b not in nx.neighbors(self.skeleton, a):
                            continue
                        if edge_types[(x, b)] != ArrowType.CIRCLE and edge_types[(z, b)] != ArrowType.CIRCLE:
                            continue
                        if edge_types[(b, a)] != ArrowType.CIRCLE:
                            continue
                        edge_types[(b, a)] = ArrowType.ARROW
                        any_changed = True
        return any_changed

    def pc_rule_1(self, edge_types):
        any_changed = False
        for a in nx.nodes(self.skeleton):
            for c in nx.non_neighbors(self.skeleton, a):
                for b in nx.common_neighbors(self.skeleton, a, c):
                    if edge_types[(a, b)] != ArrowType.ARROW:
                        continue
                    if edge_types[(c, b)] != ArrowType.CIRCLE:
                        continue
                    edge_types[(b, c)] = ArrowType.ARROW
                    edge_types[(c, b)] = ArrowType.NONE
                    any_changed = True
        return any_changed

    def pc_rule_2(self, edge_types):
        any_changed = False
        for a in nx.nodes(self.skeleton):
            for b in nx.neighbors(self.skeleton, a):
                if edge_types[(b, a)] != ArrowType.CIRCLE:
                    continue
                for c in nx.common_neighbors(self.skeleton, a, b):
                    if edge_types[(a, c)] != ArrowType.ARROW:
                        continue
                    if edge_types[(c, b)] != ArrowType.ARROW:
                        continue
                    edge_types[(a, b)] = ArrowType.ARROW
                    edge_types[(b, a)] = ArrowType.NONE
                    any_changed = True
        return any_changed

    def pc_rule_3(self, edge_types):
        any_changed = False
        for a in nx.nodes(self.skeleton):
            for b in nx.neighbors(self.skeleton, a):
                if edge_types[(b, a)] != ArrowType.CIRCLE:
                    continue
                for c in nx.common_neighbors(self.skeleton, a, b):
                    if edge_types[(c, b)] != ArrowType.ARROW:
                        continue
                    if edge_types[(c, a)] != ArrowType.CIRCLE:
                        continue
                    for d in nx.common_neighbors(self.skeleton, a, b):
                        if self.skeleton.has_edge(c, d):
                            continue
                        if edge_types[(d, b)] != ArrowType.ARROW:
                            continue
                        if edge_types[(d, a)] != ArrowType.CIRCLE:
                            continue
                        edge_types[(a, b)] = ArrowType.ARROW
                        edge_types[(b, a)] = ArrowType.NONE
                        any_changed = True
        return any_changed

    def pc_rule_4(self, edge_types):
        any_changed = False
        for a in nx.nodes(self.skeleton):
            for c in nx.neighbors(self.skeleton, a):
                if edge_types[(c, a)] != ArrowType.CIRCLE:
                    continue
                for d in nx.common_neighbors(self.skeleton, a, c):
                    if edge_types[(c, d)] != ArrowType.ARROW:
                        continue
                    for b in nx.common_neighbors(self.skeleton, a, d):
                        if edge_types[(d, b)] != ArrowType.ARROW:
                            continue
                        if edge_types[(b, a)] != ArrowType.CIRCLE:
                            continue
                        edge_types[(a, b)] = ArrowType.ARROW
                        edge_types[(b, a)] = ArrowType.NONE
                        any_changed = True
        return any_changed

    def apply_rules(self, pc=False):
        changed = True
        edge_types = dict(map(lambda edge:
                              ((edge.v_from, edge.v_to), edge.head_type),
                              self.pag))
        edge_types.update(dict(map(lambda edge:
                                   ((edge.v_to, edge.v_from), edge.tail_type),
                                   self.pag)))
        while changed:
            changed = False
            if pc:
                changed |= self.pc_rule_1(edge_types)
                changed |= self.pc_rule_2(edge_types)
                changed |= self.pc_rule_3(edge_types)
                changed |= self.pc_rule_4(edge_types)
            else:
                changed |= self.rule_1(edge_types)
                changed |= self.rule_2(edge_types)
                changed |= self.rule_3(edge_types)

        if pc:
            for pag_edge in self.pag:
                if edge_types[(pag_edge.v_from, pag_edge.v_to)] == ArrowType.CIRCLE and \
                        edge_types[(pag_edge.v_to, pag_edge.v_from)] == ArrowType.CIRCLE:
                    edge_types[(pag_edge.v_from, pag_edge.v_to)] = ArrowType.ARROW
                    edge_types[(pag_edge.v_to, pag_edge.v_from)] = ArrowType.ARROW
                    continue
                if edge_types[(pag_edge.v_from, pag_edge.v_to)] == ArrowType.CIRCLE:
                    edge_types[(pag_edge.v_from, pag_edge.v_to)] = ArrowType.NONE
                if edge_types[(pag_edge.v_to, pag_edge.v_from)] == ArrowType.CIRCLE:
                    edge_types[(pag_edge.v_to, pag_edge.v_from)] = ArrowType.NONE
        for pag_edge in self.pag:
            pag_edge.head_type = edge_types[(pag_edge.v_from, pag_edge.v_to)]
            pag_edge.tail_type = edge_types[(pag_edge.v_to, pag_edge.v_from)]

    def get_pag(self):
        return self.pag
