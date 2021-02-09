import numpy as np
import pytest

from graph_pkg.algorithm.graph_edit_distance import GED
from graph_pkg.edit_cost.edit_cost_letter import EditCostLetter
from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.node import Node
from graph_pkg.loader.loader_letter import LoaderLetter

import networkx as nx

@pytest.fixture()
def letter_graphs():
    ged = GED(EditCostLetter(1., 1., 1., 1., 'euclidean'))

    loader = LoaderLetter()
    graphs = loader.load()
    name_graphs_to_test = ['IP1_0000', 'IP1_0001']

    graphs_to_work = [graph for graph in graphs if graph.name in name_graphs_to_test]

    return graphs_to_work
    # find graph


@pytest.fixture()
def define_graphs():
    ged = GED(EditCostLetter(1., 1., 1., 1., 'manhattan'))

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

    cost = ged.compute_distance_between_graph(graph_source, graph_target)

    expected_cost = 4.

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
    assert cost == expected_cost

def test_letter_I(letter_graphs):
    epsilon = 1e-9
    cst_cost_node = 0.9
    cst_cost_edge = 2.3
    ged = GED(EditCostLetter(cst_cost_node, cst_cost_node,
                             cst_cost_edge, cst_cost_edge, 'euclidean'))

    gr1 = nx.Graph()
    gr1.add_node(0, x=1.41307, y=2.96441)
    gr1.add_node(1, x=1.47339, y=0.590991)
    gr1.add_edge(0, 1)

    gr2 = nx.Graph()
    gr2.add_node(0, x=1.51, y=3.)
    gr2.add_node(1, x=1.59592, y=0.48336)
    gr2.add_node(2, x=0.585624, y=0.486731)
    gr2.add_edge(0, 1)
    gr2.add_edge(0, 2)

    from time import time
    start_time = time()
    expected_cost = nx.algorithms.graph_edit_distance(gr1, gr2,
                                      node_subst_cost=lambda x,y: np.linalg.norm(np.array(list(x.values()))-np.array(list(y.values()))),
                                      node_ins_cost=lambda x: cst_cost_node,
                                      node_del_cost=lambda x: cst_cost_node,
                                      # edge_subst_cost=lambda x, y: 0.,
                                      edge_ins_cost=lambda x: cst_cost_edge,
                                      edge_del_cost=lambda x: cst_cost_edge)

    print(f'Computation time  NX{time()-start_time}')

    start_time = time()
    real_cost = ged.compute_distance_between_graph(letter_graphs[0], letter_graphs[1])
    print(f'Computation time {time()-start_time}')
    print(f'Expected cost: {expected_cost}')
    print(f'My cost: {real_cost}')
    assert real_cost - expected_cost < epsilon
    assert False
    # assert False

    # u = np.array([[1.41307, 2.96441], [1.47339, 0.590991]])
    # v = np.array([[1.51, 3.], [1.59592, 0.48336], [0.585624, 0.486731]])
    # n, m = u.shape[0], v.shape[0]
    # expected_C = np.zeros((n + m, n + m))
    # for i in range(n):
    #     for j in range(m):
    #         expected_C[i][j] = np.linalg.norm([u[i] - v[j]])
    #
    # expected_C[:n, m:] = np.inf
    # for i in range(n):
    #     j = i + m
    #     expected_C[i][j] = 0.9
    #
    # expected_C[n:, :m] = np.inf
    # for j in range(m):
    #     i = j + n
    #     expected_C[i][j] = 0.9
    #
    # print(expected_C)
    # print(letter_graphs[0])
    # print(letter_graphs[1])
    #
    # assert False