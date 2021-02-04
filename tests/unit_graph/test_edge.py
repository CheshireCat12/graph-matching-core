import pytest

from graph_pkg.graph.edge import Edge
from graph_pkg.graph.label.label_edge import LabelEdge

@pytest.mark.parametrize('idx_start, idx_end, label_weight',
                         [(1, 5, 0),
                          (5, 1, 0),
                          (43, 1, 3)])
def test_simple_edge(idx_start, idx_end, label_weight):
    label = LabelEdge(label_weight)
    my_edge = Edge(idx_start, idx_end, label)

    assert my_edge.idx_node_start == idx_start
    assert my_edge.idx_node_end == idx_end
    assert my_edge.weight == label
    assert str(my_edge) == f'Edge: {idx_start} --> {idx_end}, weight {label_weight}'

@pytest.fixture()
def my_edge():
    idx_start = 1
    idx_end = 5
    label = LabelEdge(0)

    edge = Edge(idx_start, idx_end, label)

    return edge
