import pytest

from graph_pkg_core.graph.edge import Edge
from graph_pkg_core.graph.label.label_edge import LabelEdge


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


@pytest.mark.parametrize('edge1, edge2, expected_result',
                         [(Edge(1, 2, LabelEdge(0)), Edge(1, 2, LabelEdge(0)), True),
                          (Edge(2, 1, LabelEdge(0)), Edge(1, 2, LabelEdge(0)), False),
                          (Edge(1, 2, LabelEdge(0)), Edge(3, 4, LabelEdge(0)), False),
                          (Edge(1, 2, LabelEdge(23)), Edge(1, 2, LabelEdge(1)), False)
                          ])
def test_edge_equality(edge1, edge2, expected_result):
    equality = edge1 == edge2

    assert equality == expected_result


@pytest.mark.parametrize('edge1, expected_edge',
                         [(Edge(1, 2, LabelEdge(0)), Edge(2, 1, LabelEdge(0))),
                          (Edge(2, 1, LabelEdge(0)), Edge(1, 2, LabelEdge(0))),
                          (Edge(3, 6, LabelEdge(3)), Edge(6, 3, LabelEdge(3))),
                          ])
def test_edge_reverse(edge1, expected_edge):
    assert edge1.reversed() == expected_edge
