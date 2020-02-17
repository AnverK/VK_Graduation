from src.causality.trees.TreeBuilder import TreeBuilder
from src.causality.trees.OperationType import OperationType
from src.causality.trees.MemoizedScores import MemoizedScores
from itertools import combinations
import networkx as nx
from time import perf_counter
import numpy as np
from math import log2


class GraphBuilder:

    def __init__(self, init='tree'):
        self.init = init
        self.cur_graph = None
        self.count_sources = 0
        self.penalty = 0
        self.base_penalty = 0
        self.scores = None

    def calc_delta_by_operation(self, edge, op_type):
        if op_type is OperationType.REVERSE_EDGE:
            u, v = edge
            d1 = self.calc_delta_by_operation(edge, OperationType.REMOVE_EDGE)
            d2 = self.calc_delta_by_operation((v, u), OperationType.ADD_EDGE)
            return d1 + d2
        affected = edge[1]
        old = list(self.cur_graph.predecessors(affected))
        old_score = self.scores.calc_score(old, affected)
        delta_penalty = 0
        if op_type is OperationType.ADD_EDGE:
            old.append(edge[0])
            if self.cur_graph.in_degree(edge[1]) == 0:
                delta_penalty = self.base_penalty
        if op_type is OperationType.REMOVE_EDGE:
            old.remove(edge[0])
            if self.cur_graph.in_degree(edge[1]) == 1:
                delta_penalty = -self.base_penalty
        new_score = self.scores.calc_score(old, affected)
        return new_score - old_score - delta_penalty

    def make_iteration(self):
        best_delta = 0
        best_action = None
        best_edge = None

        for u, v in combinations(range(self.cur_graph.number_of_nodes()), 2):
            edge = None
            if self.cur_graph.has_edge(u, v):
                edge = (u, v)
            elif self.cur_graph.has_edge(v, u):
                edge = (v, u)
            if edge is not None:
                d = self.calc_delta_by_operation(edge, OperationType.REMOVE_EDGE)
                if d > best_delta:
                    best_delta = d
                    best_action = OperationType.REMOVE_EDGE
                    best_edge = edge
                old_target_parents = set(self.cur_graph.predecessors(edge[1]))
                reachable_from_old_source = nx.single_source_shortest_path_length(self.cur_graph, edge[0])
                if len(old_target_parents.intersection(reachable_from_old_source)) != 1:
                    continue
                d = self.calc_delta_by_operation(edge, OperationType.REVERSE_EDGE)
                if d > best_delta:
                    best_delta = d
                    best_action = OperationType.REVERSE_EDGE
                    best_edge = edge
            else:
                d1 = 0 if nx.has_path(self.cur_graph, v, u) else self.calc_delta_by_operation((u, v),
                                                                                              OperationType.ADD_EDGE)
                d2 = 0 if nx.has_path(self.cur_graph, u, v) else self.calc_delta_by_operation((v, u),
                                                                                              OperationType.ADD_EDGE)
                d = max(d1, d2)
                if d > best_delta:
                    best_delta = d
                    best_action = OperationType.ADD_EDGE
                    best_edge = (u, v) if d1 >= d2 else (v, u)
        if best_delta == 0:
            return None
        if best_action == OperationType.REMOVE_EDGE:
            self.cur_graph.remove_edge(*best_edge)
        elif best_action == OperationType.ADD_EDGE:
            self.cur_graph.add_edge(*best_edge)
        else:
            self.cur_graph.remove_edge(*best_edge)
            self.cur_graph.add_edge(best_edge[1], best_edge[0])
        return best_delta, best_edge, best_action

    def build_graph(self, data):
        n_samples, n_features = data.shape
        self.scores = MemoizedScores(data)
        if self.init == 'tree':
            self.cur_graph = TreeBuilder().build_tree(data)
            assert nx.is_directed_acyclic_graph(self.cur_graph)
        start = perf_counter()
        self.base_penalty = log2(n_samples) / 2
        self.base_penalty = 0
        # self.penalty = self.base_penalty * (n_features - np.count_nonzero(self.cur_graph.in_degree))
        for i in range(100):
            res = self.make_iteration()
            if res is None:
                break
            else:
                delta, edge, action = res
                print(i, delta, edge, action)
        print(perf_counter() - start)
        return self.cur_graph
