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

    def __str__(self):
        return "({}, {}): ({}, {})".format(self.v_from, self.v_to, self.tail_type, self.head_type)
