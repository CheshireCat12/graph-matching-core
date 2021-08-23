import pytest
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.node import Node
from graph_pkg.graph.label.label_node_AIDS import LabelNodeAIDS
from graph_pkg.graph.label.label_node_NCI1 import LabelNodeNCI1
from graph_pkg.algorithm.graph_edit_distance import GED
from graph_pkg.edit_cost.edit_cost_AIDS import EditCostAIDS
from graph_pkg.edit_cost.edit_cost_NCI1 import EditCostNCI1
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

    expected_dist = 4.7
    assert round(dist, 2) == expected_dist

def test_with_deleted_node(ged):
    gr_src = Graph('gr1', 'gr1.gxl', 3)
    gr_trgt = Graph('gr2', 'gr2.gxl', 3)

    gr_src = Graph('gr1', 'gr1.gxl', 5)
    gr_src.add_node(Node(0, LabelNodeAIDS('O', 1, 1, 2., 2.)))
    gr_src.add_node(Node(1, LabelNodeAIDS('C', 1, 1, 2., 2.)))
    gr_src.add_node(Node(2, LabelNodeAIDS('O', 1, 1, 2., 2.)))
    # print(gr_src)
    gr_src.add_node(Node(3, LabelNodeAIDS('Cl', 1, 1, 2., 2.)))
    gr_src.add_node(Node(4, LabelNodeAIDS('N', 1, 1, 2.4, 2.)))

    gr_src.add_edge(Edge(0, 3, LabelEdge(0)))
    gr_src.add_edge(Edge(1, 3, LabelEdge(0)))

    gr_src.add_edge(Edge(4, 2, LabelEdge(0)))
    gr_src.add_edge(Edge(3, 4, LabelEdge(0)))

    gr_src.remove_node_by_idx(4)
    gr_src.remove_node_by_idx(3)
    print(gr_src)

    gr_trgt.add_node(Node(0, LabelNodeAIDS('H', 1, 1, 2., 2.)))
    gr_trgt.add_node(Node(1, LabelNodeAIDS('Ca', 1, 1, 2., 2.)))
    gr_trgt.add_node(Node(2, LabelNodeAIDS('C', 1, 1, 2., 2.)))

    gr_trgt.add_edge(Edge(0, 1, LabelEdge(0)))
    gr_trgt.add_edge(Edge(1, 2, LabelEdge(0)))
    gr_trgt.add_edge(Edge(2, 0, LabelEdge(0)))


    dist = ged.compute_edit_distance(gr_src, gr_trgt)

    print(ged.C.base)
    print(ged.C_star.base)

    expected_dist = 4.7
    print(dist)
    assert round(dist, 2) == expected_dist

def test_didactic_full():
    ged = GED(EditCostNCI1(1.0, 1.0, 1.0, 1.0, 'dirac', alpha=0.5))
    # ged = GED(EditCostNCI1(0.5, 0.5, 0.5, 0.5, 'dirac', alpha=1.0))
    gr_src = Graph('gr1', 'gr1.gxl', 14)
    gr_trgt = Graph('gr2', 'gr2.gxl', 15)

    gr_src.add_node(Node(0, LabelNodeNCI1(3)))
    gr_src.add_node(Node(1, LabelNodeNCI1(1)))
    gr_src.add_node(Node(2, LabelNodeNCI1(2)))
    gr_src.add_node(Node(3, LabelNodeNCI1(2)))
    gr_src.add_node(Node(4, LabelNodeNCI1(2)))
    gr_src.add_node(Node(5, LabelNodeNCI1(2)))
    gr_src.add_node(Node(6, LabelNodeNCI1(1)))
    gr_src.add_node(Node(7, LabelNodeNCI1(1)))
    gr_src.add_node(Node(8, LabelNodeNCI1(2)))
    gr_src.add_node(Node(9, LabelNodeNCI1(2)))
    gr_src.add_node(Node(10, LabelNodeNCI1(2)))
    gr_src.add_node(Node(11, LabelNodeNCI1(3)))
    gr_src.add_node(Node(12, LabelNodeNCI1(3)))
    gr_src.add_node(Node(13, LabelNodeNCI1(3)))

    gr_src.add_edge(Edge(0, 1, LabelEdge(0)))
    gr_src.add_edge(Edge(1, 2, LabelEdge(0)))
    gr_src.add_edge(Edge(1, 3, LabelEdge(0)))
    gr_src.add_edge(Edge(2, 5, LabelEdge(0)))
    gr_src.add_edge(Edge(3, 4, LabelEdge(0)))
    gr_src.add_edge(Edge(4, 6, LabelEdge(0)))
    gr_src.add_edge(Edge(5, 6, LabelEdge(0)))
    gr_src.add_edge(Edge(6, 7, LabelEdge(0)))
    gr_src.add_edge(Edge(7, 8, LabelEdge(0)))
    gr_src.add_edge(Edge(7, 9, LabelEdge(0)))
    gr_src.add_edge(Edge(8, 10, LabelEdge(0)))
    gr_src.add_edge(Edge(9, 10, LabelEdge(0)))
    gr_src.add_edge(Edge(8, 11, LabelEdge(0)))
    gr_src.add_edge(Edge(8, 12, LabelEdge(0)))
    gr_src.add_edge(Edge(9, 13, LabelEdge(0)))

    gr_trgt.add_node(Node(0, LabelNodeNCI1(3)))
    gr_trgt.add_node(Node(1, LabelNodeNCI1(1)))
    gr_trgt.add_node(Node(2, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(3, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(4, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(5, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(6, LabelNodeNCI1(1)))
    gr_trgt.add_node(Node(7, LabelNodeNCI1(1)))
    gr_trgt.add_node(Node(8, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(9, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(10, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(11, LabelNodeNCI1(3)))
    gr_trgt.add_node(Node(12, LabelNodeNCI1(3)))
    gr_trgt.add_node(Node(13, LabelNodeNCI1(3)))
    gr_trgt.add_node(Node(14, LabelNodeNCI1(3)))


    gr_trgt.add_edge(Edge(0, 1, LabelEdge(0)))
    gr_trgt.add_edge(Edge(1, 2, LabelEdge(0)))
    gr_trgt.add_edge(Edge(1, 3, LabelEdge(0)))
    gr_trgt.add_edge(Edge(2, 5, LabelEdge(0)))
    gr_trgt.add_edge(Edge(3, 4, LabelEdge(0)))
    gr_trgt.add_edge(Edge(4, 6, LabelEdge(0)))
    gr_trgt.add_edge(Edge(5, 6, LabelEdge(0)))
    gr_trgt.add_edge(Edge(6, 7, LabelEdge(0)))
    gr_trgt.add_edge(Edge(7, 8, LabelEdge(0)))
    gr_trgt.add_edge(Edge(7, 9, LabelEdge(0)))
    gr_trgt.add_edge(Edge(8, 10, LabelEdge(0)))
    gr_trgt.add_edge(Edge(9, 10, LabelEdge(0)))
    gr_trgt.add_edge(Edge(8, 11, LabelEdge(0)))
    gr_trgt.add_edge(Edge(8, 12, LabelEdge(0)))
    gr_trgt.add_edge(Edge(9, 13, LabelEdge(0)))

    dist = ged.compute_edit_distance(gr_src, gr_trgt)
    C = np.array(ged.C)
    C_star = np.array(ged.C_star)
    phi = np.array(ged.phi)

    C[C > 99999] = 9999
    C_star[C_star > 99999] = 9999

    np.savetxt('orC_full.csv', C, delimiter=',', fmt='%f')
    np.savetxt('orC_star_full.csv', C_star, delimiter=',', fmt='%f')
    np.savetxt('orPhi_full.csv', phi, delimiter=',', fmt='%f')
    # print(C)
    expected_dist = 4.7
    print(dist)
    assert False

def test_didactic_80():
    ged = GED(EditCostNCI1(1.0, 1.0, 1.0, 1.0, 'dirac', alpha=0.5))
    gr_src = Graph('gr1', 'gr1.gxl', 14)
    gr_trgt = Graph('gr2', 'gr2.gxl', 15)

    gr_src.add_node(Node(0, LabelNodeNCI1(3)))
    gr_src.add_node(Node(1, LabelNodeNCI1(1)))
    gr_src.add_node(Node(2, LabelNodeNCI1(2)))
    gr_src.add_node(Node(3, LabelNodeNCI1(2)))
    gr_src.add_node(Node(4, LabelNodeNCI1(2)))
    gr_src.add_node(Node(5, LabelNodeNCI1(2)))
    gr_src.add_node(Node(6, LabelNodeNCI1(1)))
    gr_src.add_node(Node(7, LabelNodeNCI1(1)))
    gr_src.add_node(Node(8, LabelNodeNCI1(2)))
    gr_src.add_node(Node(9, LabelNodeNCI1(2)))
    gr_src.add_node(Node(10, LabelNodeNCI1(2)))

    gr_src.add_edge(Edge(0, 1, LabelEdge(0)))
    gr_src.add_edge(Edge(1, 2, LabelEdge(0)))
    gr_src.add_edge(Edge(1, 3, LabelEdge(0)))
    gr_src.add_edge(Edge(2, 5, LabelEdge(0)))
    gr_src.add_edge(Edge(3, 4, LabelEdge(0)))
    gr_src.add_edge(Edge(4, 6, LabelEdge(0)))
    gr_src.add_edge(Edge(5, 6, LabelEdge(0)))
    gr_src.add_edge(Edge(6, 7, LabelEdge(0)))
    gr_src.add_edge(Edge(7, 8, LabelEdge(0)))
    gr_src.add_edge(Edge(7, 9, LabelEdge(0)))
    gr_src.add_edge(Edge(8, 10, LabelEdge(0)))
    gr_src.add_edge(Edge(9, 10, LabelEdge(0)))

    gr_trgt.add_node(Node(0, LabelNodeNCI1(3)))
    gr_trgt.add_node(Node(1, LabelNodeNCI1(1)))
    gr_trgt.add_node(Node(2, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(3, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(4, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(5, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(6, LabelNodeNCI1(1)))
    gr_trgt.add_node(Node(7, LabelNodeNCI1(1)))
    gr_trgt.add_node(Node(8, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(9, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(10, LabelNodeNCI1(2)))
    gr_trgt.add_node(Node(11, LabelNodeNCI1(3)))

    gr_trgt.add_edge(Edge(0, 1, LabelEdge(0)))
    gr_trgt.add_edge(Edge(1, 2, LabelEdge(0)))
    gr_trgt.add_edge(Edge(1, 3, LabelEdge(0)))
    gr_trgt.add_edge(Edge(2, 5, LabelEdge(0)))
    gr_trgt.add_edge(Edge(3, 4, LabelEdge(0)))
    gr_trgt.add_edge(Edge(4, 6, LabelEdge(0)))
    gr_trgt.add_edge(Edge(5, 6, LabelEdge(0)))
    gr_trgt.add_edge(Edge(6, 7, LabelEdge(0)))
    gr_trgt.add_edge(Edge(7, 8, LabelEdge(0)))
    gr_trgt.add_edge(Edge(7, 9, LabelEdge(0)))
    gr_trgt.add_edge(Edge(8, 10, LabelEdge(0)))
    gr_trgt.add_edge(Edge(9, 10, LabelEdge(0)))
    gr_trgt.add_edge(Edge(8, 11, LabelEdge(0)))


    dist = ged.compute_edit_distance(gr_src, gr_trgt)
    C = np.array(ged.C)
    C_star = np.array(ged.C_star)
    phi = np.array(ged.phi)

    C[C > 99999] = 9999
    C_star[C_star > 99999] = 9999

    np.savetxt('orC_80.csv', C, delimiter=',', fmt='%f')
    np.savetxt('orC_star_80.csv', C_star, delimiter=',', fmt='%f')
    np.savetxt('orPhi_80.csv', phi, delimiter=',', fmt='%f')
    # print(ged.C.base)
    # print(ged.C_star.base)

    expected_dist = 4.7
    print(dist)
    assert False
