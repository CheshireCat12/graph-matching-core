import pytest
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.node import Node
from graph_pkg.graph.label.label_node_AIDS import LabelNodeAIDS
from graph_pkg.algorithm.graph_edit_distance import GED
from graph_pkg.edit_cost.edit_cost_AIDS import EditCostAIDS
from graph_pkg.graph.edge import Edge
from graph_pkg.graph.label.label_edge import LabelEdge

import numpy as np

@pytest.fixture()
def ged():
    graph_edit_distance = GED(EditCostAIDS(1.1, 1.1, 0.1, 0.1, 'dirac'))
    return graph_edit_distance


def test_1_node(ged):
    gr_src = Graph('gr1', 'gr1.gxl', 1)
    gr_trgt = Graph('gr2', 'gr2.gxl', 1)

    gr_src.add_node(Node(0, LabelNodeAIDS('C', 1, 1, 2., 2.)))

    gr_trgt.add_node(Node(0, LabelNodeAIDS('O', 1, 1, 2., 2.)))

    dist = ged.compute_edit_distance(gr_src, gr_trgt)

    expected_dist = 2.2
    print(dist)
    print(ged.C.base)
    print(ged.C_star.base)
    print(ged.phi.base)
    assert dist == expected_dist


def test_1_node_same(ged):
    gr_src = Graph('gr1', 'gr1.gxl', 1)
    gr_trgt = Graph('gr2', 'gr2.gxl', 1)

    gr_src.add_node(Node(0, LabelNodeAIDS('C', 1, 1, 2., 2.)))

    gr_trgt.add_node(Node(0, LabelNodeAIDS('C', 1, 1, 2., 2.)))

    dist = ged.compute_edit_distance(gr_src, gr_trgt)

    expected_dist = 0.
    assert dist == expected_dist


def test_2_node(ged):
    gr_src = Graph('gr1', 'gr1.gxl', 2)
    gr_trgt = Graph('gr2', 'gr2.gxl', 2)

    gr_src.add_node(Node(0, LabelNodeAIDS('C', 1, 1, 2., 2.)))
    gr_src.add_node(Node(1, LabelNodeAIDS('O', 1, 1, 2., 2.)))

    gr_trgt.add_node(Node(0, LabelNodeAIDS('C', 1, 1, 2., 2.)))
    gr_trgt.add_node(Node(1, LabelNodeAIDS('C', 1, 1, 2., 2.)))

    dist = ged.compute_edit_distance(gr_src, gr_trgt)

    np_C = np.asarray(ged.C)
    expected_C = np.array([[0., 0, 1.1, np.inf],
                           [2.2, 2.2, np.inf, 1.1],
                           [1.1, np.inf, 0., 0.],
                           [np.inf, 1.1, 0., 0.]])
    assert np.array_equal(np_C, expected_C)

    expected_dist = 2.2
    assert dist == expected_dist



def test_3_nodes(ged):
    gr_src = Graph('gr1', 'gr1.gxl', 3)
    gr_trgt = Graph('gr2', 'gr2.gxl', 3)

    gr_src.add_node(Node(0, LabelNodeAIDS('O', 1, 1, 2., 2.)))
    gr_src.add_node(Node(1, LabelNodeAIDS('C', 1, 1, 2., 2.)))
    gr_src.add_node(Node(2, LabelNodeAIDS('O', 1, 1, 2., 2.)))

    gr_trgt.add_node(Node(0, LabelNodeAIDS('O', 1, 1, 2., 2.)))
    gr_trgt.add_node(Node(1, LabelNodeAIDS('C', 1, 1, 2., 2.)))
    gr_trgt.add_node(Node(2, LabelNodeAIDS('O', 1, 1, 2., 2.)))

    gr_trgt.add_edge(Edge(0, 1, LabelEdge(0)))
    gr_trgt.add_edge(Edge(1, 2, LabelEdge(0)))

    dist = ged.compute_edit_distance(gr_src, gr_trgt)

    print(ged.C.base)
    print(ged.C_star.base)

    expected_dist = 0.2
    assert dist == expected_dist

def test_3_nodes_1_node(ged):
    gr_src = Graph('gr1', 'gr1.gxl', 3)
    gr_trgt = Graph('gr2', 'gr2.gxl', 1)

    gr_src.add_node(Node(0, LabelNodeAIDS('O', 1, 1, 2., 2.)))
    gr_src.add_node(Node(1, LabelNodeAIDS('C', 1, 1, 2., 2.)))
    gr_src.add_node(Node(2, LabelNodeAIDS('O', 1, 1, 2., 2.)))

    gr_src.add_edge(Edge(0, 1, LabelEdge(0)))
    gr_src.add_edge(Edge(1, 2, LabelEdge(0)))

    gr_trgt.add_node(Node(0, LabelNodeAIDS('H', 1, 1, 2., 2.)))


    dist = ged.compute_edit_distance(gr_src, gr_trgt)

    print(ged.C.base)
    print(ged.C_star.base)

    expected_dist = 4.6
    assert round(dist, 2) == expected_dist


def test_3_nodes_3_nodes(ged):
    gr_src = Graph('gr1', 'gr1.gxl', 3)
    gr_trgt = Graph('gr2', 'gr2.gxl', 3)

    gr_src.add_node(Node(0, LabelNodeAIDS('O', 1, 1, 2., 2.)))
    gr_src.add_node(Node(1, LabelNodeAIDS('C', 1, 1, 2., 2.)))
    gr_src.add_node(Node(2, LabelNodeAIDS('O', 1, 1, 2., 2.)))

    gr_trgt.add_node(Node(0, LabelNodeAIDS('H', 1, 1, 2., 2.)))
    gr_trgt.add_node(Node(1, LabelNodeAIDS('Ca', 1, 1, 2., 2.)))
    gr_trgt.add_node(Node(2, LabelNodeAIDS('C', 1, 1, 2., 2.)))

    gr_trgt.add_edge(Edge(0, 1, LabelEdge(0)))
    gr_trgt.add_edge(Edge(1, 2, LabelEdge(0)))
    gr_trgt.add_edge(Edge(2, 0, LabelEdge(0)))


    dist = ged.compute_edit_distance(gr_src, gr_trgt)

    print(ged.C.base)
    print(ged.C_star.base)

    expected_dist = 4.6
    assert round(dist, 2) == expected_dist
