import pickle
import sys

import numpy as np
import pytest

from graph_pkg.algorithm.graph_edit_distance import GED
from graph_pkg.edit_cost.edit_cost_AIDS import EditCostAIDS
from graph_pkg.edit_cost.edit_cost_letter import EditCostLetter
from graph_pkg.edit_cost.edit_cost_mutagenicity import EditCostMutagenicity
from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.node import Node
from graph_pkg.loader.loader_AIDS import LoaderAIDS
from graph_pkg.loader.loader_letter import LoaderLetter
from graph_pkg.loader.loader_mutagenicity import LoaderMutagenicity


loader = LoaderLetter('HIGH')
letter_graphs = loader.load()

@pytest.fixture()
def aids_graphs():
    loader = LoaderAIDS()
    graphs = loader.load()
    return graphs


@pytest.fixture()
def mutagenicity_graphs():
    loader = LoaderMutagenicity()
    graphs = loader.load()
    return graphs


@pytest.fixture()
def define_graphs():

    n, m = 4, 3
    graph_source = Graph('gr_source', 'gr_source.gxl', n)
    graph_target = Graph('gr_target', 'gr_targe.gxl', m)

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

    return graph_source, graph_target


def test_simple_alpha(define_graphs):
    graph_source, graph_target = define_graphs

    # edit_cost = EditCostLetter(1., 1., 1., 1., 'euclidean')
    edit_cost_alpha = EditCostLetter(1., 1., 1., 1., 'euclidean', alpha=0.5)

    # ged = GED(edit_cost)
    ged_alpha = GED(edit_cost_alpha)

    # cost = ged.compute_edit_distance(graph_source, graph_target)
    cost_alpha = ged_alpha.compute_edit_distance(graph_source, graph_target)

    expected_cost_alpha = 2.

    expected_C = np.array([[2., 1., 1., 1., np.inf, np.inf, np.inf],
                           [1., 0., 0., np.inf, 1., np.inf, np.inf],
                           [2., 1., 1., np.inf, np.inf, 1., np.inf],
                           [0., 1., 1., np.inf, np.inf, np.inf, 1.],
                           [1., np.inf, np.inf, 0., 0., 0., 0.],
                           [np.inf, 1., np.inf, 0., 0., 0., 0.],
                           [np.inf, np.inf, 1., 0., 0., 0., 0.]])
    expected_C_alpha = expected_C / 2

    expected_C_star = np.array([[2., 2., 1., 2., np.inf, np.inf, np.inf],
                                [3., 1., 2., np.inf, 4., np.inf, np.inf],
                                [3., 1., 2., np.inf, np.inf, 3., np.inf],
                                [1., 1., 2., np.inf, np.inf, np.inf, 3.],
                                [2., np.inf, np.inf, 0., 0., 0., 0.],
                                [np.inf, 3., np.inf, 0., 0., 0., 0.],
                                [np.inf, np.inf, 2., 0., 0., 0., 0.]])
    expected_C_star_alpha = expected_C_star / 2


    print(ged_alpha.C.base)

    assert np.array_equal(np.asarray(ged_alpha.C), expected_C_alpha)
    assert np.array_equal(np.asarray(ged_alpha.C_star), expected_C_star_alpha)
    assert cost_alpha == expected_cost_alpha

# @pytest.mark.skip()
def test_alpha_0_25(define_graphs):
    graph_source, graph_target = define_graphs

    edit_cost = EditCostLetter(1., 1., 1., 1., 'euclidean')
    edit_cost_alpha = EditCostLetter(1., 1., 1., 1., 'euclidean', alpha=0.25)
    ged = GED(edit_cost)
    ged_alpha = GED(edit_cost_alpha)

    cost = ged.compute_edit_distance(graph_source, graph_target)
    cost_alpha = ged_alpha.compute_edit_distance(graph_source, graph_target)

    expected_cost_alpha = 2.

    expected_C = np.array([[2., 1., 1., 1., np.inf, np.inf, np.inf],
                           [1., 0., 0., np.inf, 1., np.inf, np.inf],
                           [2., 1., 1., np.inf, np.inf, 1., np.inf],
                           [0., 1., 1., np.inf, np.inf, np.inf, 1.],
                           [1., np.inf, np.inf, 0., 0., 0., 0.],
                           [np.inf, 1., np.inf, 0., 0., 0., 0.],
                           [np.inf, np.inf, 1., 0., 0., 0., 0.]])
    expected_C_alpha = expected_C / 4

    expected_C_star_alpha = np.array([[0.5, 1., 0.25, 1., np.inf, np.inf, np.inf],
                                      [1.75, 0.75, 1.5, np.inf, 2.5, np.inf, np.inf],
                                      [1.25, 0.25, 1., np.inf, np.inf, 1.75, np.inf],
                                      [0.75, 0.25, 1., np.inf, np.inf, np.inf, 1.75],
                                      [1., np.inf, np.inf, 0., 0., 0., 0.],
                                      [np.inf, 1.75, np.inf, 0., 0., 0., 0.],
                                      [np.inf, np.inf, 1., 0., 0., 0., 0.]])
    # expected_C_star_alpha = expected_C_star / 2
    print(ged_alpha.C_star.base)

    assert np.array_equal(np.asarray(ged_alpha.C), expected_C_alpha)
    assert np.array_equal(np.asarray(ged_alpha.C_star), expected_C_star_alpha)
    assert cost_alpha == expected_cost_alpha


# @pytest.mark.skip()
@pytest.mark.parametrize('graphs, graph_source_target, accuracy',
                         [(letter_graphs, ['AP1_0000', 'AP1_0001'], 1e-6),
                          (letter_graphs, ['IP1_0000', 'IP1_0001'], 1e-6),
                          (letter_graphs, ['AP1_0100', 'IP1_0000'], 1e-6),
                          (letter_graphs, ['HP1_0100', 'WP1_0010'], 1e-6),
                          (letter_graphs, ['XP1_0005', 'KP1_0023'], 1e-6),
                          (letter_graphs, ['EP1_0120', 'LP1_0099'], 1e-6),
                          (letter_graphs, ['MP1_0019', 'FP1_0083'], 1e-6),
                          ])
def test_letter_alpha_0_5(graphs, graph_source_target, accuracy):
    graph_source, graph_target = [graph for graph in graphs if graph.name in graph_source_target]

    cst_cost_node = 0.9
    cst_cost_edge = 2.3
    edit_cost = EditCostLetter(cst_cost_node, cst_cost_node, cst_cost_edge, cst_cost_edge, 'euclidean')
    edit_cost_alpha = EditCostLetter(cst_cost_node, cst_cost_node, cst_cost_edge, cst_cost_edge, 'euclidean', alpha=0.5)
    ged = GED(edit_cost)
    ged_alpha = GED(edit_cost_alpha)

    results = ged_alpha.compute_edit_distance(graph_source, graph_target)
    expected = ged.compute_edit_distance(graph_source, graph_target) / 2.

    assert (results - expected) < accuracy


@pytest.mark.parametrize('graph_name_source, graph_name_target',
                         [(['molid600779', 'molid409962']),
                          (['molid624151', 'molid633011']),
                          (['molid633011', 'molid624151']),
                          (['molid660165', 'molid645098']),
                          (['molid645098', 'molid660165']),
                          ])
def test_aids_alpha(aids_graphs, graph_name_source, graph_name_target):
    graph_source = [graph for graph in aids_graphs if graph.name == graph_name_source][0]
    graph_target = [graph for graph in aids_graphs if graph.name == graph_name_target][0]

    cst_cost_node = 1.1
    cst_cost_edge = 0.1
    edit_cost = EditCostAIDS(cst_cost_node, cst_cost_node,
                             cst_cost_edge, cst_cost_edge, 'dirac')
    edit_cost_alpha = EditCostAIDS(cst_cost_node, cst_cost_node,
                                   cst_cost_edge, cst_cost_edge, 'dirac', alpha=0.5)
    ged = GED(edit_cost)
    ged_alpha = GED(edit_cost_alpha)

    results = ged_alpha.compute_edit_distance(graph_source, graph_target)
    expected = ged.compute_edit_distance(graph_source, graph_target) / 2.

    assert results == expected

@pytest.mark.parametrize('graph_name_source_target',
                         [
                             (['molecule_2767', 'molecule_2769']),
                             (['molecule_2769', 'molecule_2767']),
                             (['molecule_1897', 'molecule_1349']),
                             (['molecule_1897', 'molecule_1051']),
                         ])
def test_mutagenicity_alpha(mutagenicity_graphs, graph_name_source_target):
    graph_name_source, graph_name_target = graph_name_source_target
    graph_source = [graph for graph in mutagenicity_graphs if graph.name == graph_name_source][0]
    graph_target = [graph for graph in mutagenicity_graphs if graph.name == graph_name_target][0]

    cst_cost_node = 11.0
    cst_cost_edge = 1.1
    edit_cost = EditCostMutagenicity(cst_cost_node, cst_cost_node,
                             cst_cost_edge, cst_cost_edge, 'dirac')
    edit_cost_alpha = EditCostMutagenicity(cst_cost_node, cst_cost_node,
                                   cst_cost_edge, cst_cost_edge, 'dirac', alpha=0.5)
    ged = GED(edit_cost)
    ged_alpha = GED(edit_cost_alpha)

    results = ged_alpha.compute_edit_distance(graph_source, graph_target)
    expected = ged.compute_edit_distance(graph_source, graph_target) / 2.

    assert results == expected
