from enum import Enum


class ArrowType(Enum):
    CIRCLE = 1
    ARROW = 2
    NONE = 3


class PagEdge:
    def __init__(self, v_from, v_to, tail, head):
        self.v_from = v_from
        self.v_to = v_to
        self.tail_type = ArrowType(tail)
        self.head_type = ArrowType(head)

    @staticmethod
    def _orient_pag(arrow_type):
        if arrow_type == ArrowType.CIRCLE:
            return 'odot'
        elif arrow_type == ArrowType.ARROW:
            return 'normal'
        else:
            assert arrow_type == ArrowType.NONE
            return 'none'

    @staticmethod
    def to_regular_edge(pag_edge):
        return (pag_edge.v_from, pag_edge.v_to)

    def __str__(self):
        return "({}, {}): ({}, {})".format(self.v_from, self.v_to, self.tail_type, self.head_type)

    def __eq__(self, other):
        if not isinstance(other, PagEdge):
            return False
        if self.v_from == other.v_from and self.v_to == other.v_to:
            return self.head_type == other.head_type and self.tail_type == other.tail_type
        if self.v_from == other.v_to and self.v_to == other.v_from:
            return self.head_type == other.tail_type and self.tail_type == other.head_type
        return False
