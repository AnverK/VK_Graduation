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
