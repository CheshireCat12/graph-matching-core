import numpy as np
import pytest

from graph_pkg.algorithm.graph_edit_distance import GED
from graph_pkg.edit_cost.edit_cost_letter_minkowski import EditCostLetterMinkowski
from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.node import Node


@pytest.fixture()
def define_graphs():
    ged = GED(EditCostLetterMinkowski(1))

    n, m = 4, 3
    graph_source = Graph('gr_source', n)
    graph_target = Graph('gr_target', m)

    # Init graph source: add nodes and edges
    graph_source.add_node(Node(0, LabelNodeLetter(1, 0)))
    graph_source.add_node(Node(1, LabelNodeLetter(2, 0)))
    graph_source.add_node(Node(2, LabelNodeLetter(1, 0)))
    graph_source.add_node(Node(3, LabelNodeLetter(3, 0)))

    graph_source.add_edge(Edge(0, 1, LabelEdge(0)))
    graph_source.add_edge(Edge(1, 2, LabelEdge(0)))
    graph_source.add_edge(Edge(1, 3, LabelEdge(0)))
    graph_source.add_edge(Edge(2, 3, LabelEdge(0)))

    # Init graph target: add nodes and edges
    graph_target.add_node(Node(0, LabelNodeLetter(3, 0)))
    graph_target.add_node(Node(1, LabelNodeLetter(2, 0)))
    graph_target.add_node(Node(2, LabelNodeLetter(2, 0)))

    graph_target.add_edge(Edge(0, 1, LabelEdge(0)))
    graph_target.add_edge(Edge(1, 2, LabelEdge(0)))

    return ged, graph_source, graph_target


def test_simple_ged(define_graphs):
    ged, graph_source, graph_target = define_graphs

    ged.compute_distance_between_graph(graph_source, graph_target)

    expected_C = np.array([[2., 1., 1., 1., np.inf, np.inf, np.inf],
                           [1., 0., 0., np.inf, 1., np.inf, np.inf],
                           [2., 1., 1., np.inf, np.inf, 1., np.inf],
                           [0., 1., 1., np.inf, np.inf, np.inf, 1.],
                           [1., np.inf, np.inf, 0., 0., 0., 0.],
                           [np.inf, 1., np.inf, 0., 0., 0., 0.],
                           [np.inf, np.inf, 1., 0., 0., 0., 0.]])

    expected_C_star = np.array([[2., 2., 1., 2., np.inf, np.inf, np.inf],
                                [3., 1., 2., np.inf, 4., np.inf, np.inf],
                                [3., 1., 2., np.inf, np.inf, 3., np.inf],
                                [1., 1., 2., np.inf, np.inf, np.inf, 3.],
                                [2., np.inf, np.inf, 0., 0., 0., 0.],
                                [np.inf, 3., np.inf, 0., 0., 0., 0.],
                                [np.inf, np.inf, 2., 0., 0., 0., 0.]])
    print('c')
    print(ged.C.base)
    print('c_star')
    print(ged.C_star.base)

    assert np.array_equal(np.asarray(ged.C), expected_C)
    assert np.array_equal(np.asarray(ged.C_star), expected_C_star)
    assert len(graph_source) == 4
    assert len(graph_target) == 3
