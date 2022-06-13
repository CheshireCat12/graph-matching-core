import os
from itertools import product

import numpy as np
import pytest

from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.graph.edge import Edge
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.graph.label.label_edge import LabelEdge
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.node import Node
from graph_pkg_core.loader.loader_vector import LoaderVector

FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           '../test_data/proteins_old')


@pytest.fixture()
def test_graphs():
    loader = LoaderVector(FOLDER_DATA)
    graphs = loader.load()

    return graphs


@pytest.fixture()
def defined_graph():
    ged = GED(EditCostVector(1., 1., 1., 1., 'euclidean'))

    n, m = 4, 3
    graph_source = Graph('gr_source', 'gr_source.gxl', n)
    graph_target = Graph('gr_target', 'gr_targe.gxl', m)

    # Init graph source: add nodes and edges
    graph_source.add_node(Node(0, LabelNodeVector(np.array([1.]))))
    graph_source.add_node(Node(1, LabelNodeVector(np.array([2.]))))
    graph_source.add_node(Node(2, LabelNodeVector(np.array([1.]))))
    graph_source.add_node(Node(3, LabelNodeVector(np.array([3.]))))

    graph_source.add_edge(Edge(0, 1, LabelEdge(0)))
    graph_source.add_edge(Edge(1, 2, LabelEdge(0)))
    graph_source.add_edge(Edge(1, 3, LabelEdge(0)))
    graph_source.add_edge(Edge(2, 3, LabelEdge(0)))

    # Init graph target: add nodes and edges
    graph_target.add_node(Node(0, LabelNodeVector(np.array([3.]))))
    graph_target.add_node(Node(1, LabelNodeVector(np.array([2.]))))
    graph_target.add_node(Node(2, LabelNodeVector(np.array([2.]))))
    graph_target.add_edge(Edge(0, 1, LabelEdge(0)))
    graph_target.add_edge(Edge(1, 2, LabelEdge(0)))

    return ged, graph_source, graph_target


def test_simple_ged(defined_graph):
    ged, graph_source, graph_target = defined_graph

    cost = ged.compute_edit_distance(graph_source, graph_target)

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
    # np.set_printoptions(precision=2)
    # print('c')
    # print(ged.C.base)
    # print('c_star')
    # print(ged.C_star.base)

    assert np.array_equal(np.asarray(ged.C), expected_C)
    assert np.array_equal(np.asarray(ged.C_star), expected_C_star)
    assert len(graph_source) == 4
    assert len(graph_target) == 3
    assert cost == expected_cost


def test_ged_same_graph(defined_graph):
    ged, graph_source, graph_target = defined_graph

    cost = ged.compute_edit_distance(graph_source, graph_source)

    expected_cost = 0.

    assert cost == expected_cost


def test_heuristic_size(defined_graph):
    ged, graph_source, graph_target = defined_graph

    cost_1 = ged.compute_edit_distance(graph_source, graph_target, heuristic=True)
    cost_2 = ged.compute_edit_distance(graph_target, graph_source, heuristic=True)

    assert cost_1 == cost_2


@pytest.mark.parametrize('idx_tr, idx_te, expected_dist',
                         [
                             ([0, 1, 2, 3],
                              [4, 5, 6],
                              np.array([[22.4, 43.6, 98.6],
                                        [44.0391919, 43.44507935, 84.4],
                                        [24.45685425, 40.4, 96.2],
                                        [26.3254834, 42.2, 97.6]]))

                         ])
def test_real_graphs(test_graphs, idx_tr, idx_te, expected_dist):
    ged = GED(EditCostVector(1., 1., 1., 1., 'euclidean', alpha=0.8))
    dists = []
    for idx1, idx2 in product(idx_tr, idx_te):
        gr_1 = test_graphs[idx1]
        gr_2 = test_graphs[idx2]
        dists.append(ged.compute_edit_distance(gr_1, gr_2, heuristic=True))

    results = np.array(dists).reshape(len(idx_tr), len(idx_te))

    assert np.linalg.norm(results - expected_dist) < 1e-8
