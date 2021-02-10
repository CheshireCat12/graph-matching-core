import numpy as np
import pickle
import pytest

from graph_pkg.algorithm.graph_edit_distance import GED
from graph_pkg.edit_cost.edit_cost_letter import EditCostLetter
from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.node import Node
from graph_pkg.loader.loader_letter import LoaderLetter
from graph_pkg.loader.loader_AIDS import LoaderAIDS
from graph_pkg.edit_cost.edit_cost_AIDS import EditCostAIDS

import networkx as nx

@pytest.fixture()
def letter_graphs():

    loader = LoaderLetter('HIGH')
    graphs = loader.load()
    return graphs
    # find graph


@pytest.fixture()
def aids_graphs():
    loader = LoaderAIDS()
    graphs = loader.load()
    return graphs


@pytest.fixture()
def dataframe_letter():
    with open('./data/goal/anthony_ged_dist_mat_alpha_node_cost0.9_edge_cost2.3.pkl', 'rb') as file:
        df = pickle.load(file)

    return df


@pytest.fixture()
def dataframe_aids():
    with open('./data/goal/anthony_ged_dist_mat_alpha_node_cost1.1_edge_cost0.1.pkl', 'rb') as file:
        df = pickle.load(file)

    return df
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
    name_graphs_to_test = ['IP1_0000', 'IP1_0001']

    letter_graphs = [graph for graph in letter_graphs if graph.name in name_graphs_to_test]
    epsilon = 1e-14
    cst_cost_node = 0.9
    cst_cost_edge = 2.3
    ged = GED(EditCostLetter(cst_cost_node, cst_cost_node,
                             cst_cost_edge, cst_cost_edge, 'euclidean'))

    gr1 = nx.Graph()
    gr1.add_node(0, x=1.66831, y=2.93501)
    gr1.add_node(1, x=1.34125, y=0.290566)
    gr1.add_edge(0, 1)

    gr2 = nx.Graph()
    gr2.add_node(0, x=1.18827, y=3.20702)

    from time import time
    start_time = time()
    expected_cost = nx.algorithms.graph_edit_distance(gr1, gr2,
                                      node_subst_cost=lambda x, y: np.linalg.norm(np.array(list(x.values()))-np.array(list(y.values()))),
                                      node_ins_cost=lambda x: cst_cost_node,
                                      node_del_cost=lambda x: cst_cost_node,
                                      # edge_subst_cost=lambda x, y: 0.,
                                      edge_ins_cost=lambda x: cst_cost_edge,
                                      edge_del_cost=lambda x: cst_cost_edge)

    print(f'Computation time NX {(time()-start_time) * 1000}')

    start_time = time()
    real_cost = ged.compute_edit_distance(letter_graphs[0],
                                          letter_graphs[1])
    print(f'Computation time {(time()-start_time) * 1000}')
    print(f'Expected cost: {expected_cost}')
    print(f'My cost:       {real_cost}')
    assert abs(real_cost - expected_cost) < epsilon
    # assert False
    # assert False

@pytest.mark.skip(reason='I have to had the expected accuracy')
@pytest.mark.parametrize('graph_source_target',
                         [(['AP1_0000', 'AP1_0001']),
                          (['IP1_0000', 'IP1_0001']),
                          (['AP1_0100', 'IP1_0000']),
                          (['HP1_0100', 'WP1_0010']),
                          (['XP1_0005', 'KP1_0023']),
                          (['EP1_0120', 'LP1_0099']),
                          (['MP1_0019', 'FP1_0083']),
                          ])
def test_with_verified_data(letter_graphs, dataframe_letter, graph_source_target):
    gr_name_src, gr_name_trgt = [name[0] + '/' + name for name in graph_source_target]
    graph_source, graph_target = [graph for graph in letter_graphs if graph.name in graph_source_target]

    cst_cost_node = 0.9
    cst_cost_edge = 2.3
    ged = GED(EditCostLetter(cst_cost_node, cst_cost_node,
                             cst_cost_edge, cst_cost_edge, 'euclidean'))

    results = ged.compute_edit_distance(graph_source, graph_target)
    expected = dataframe_letter.loc[gr_name_src, gr_name_trgt]

    print(f'###### diff {results - expected}')
    assert results == expected


@pytest.mark.parametrize('graph_name_source, graph_name_target, gr_name_src, gr_name_trgt',
                         [(['molid624151', 'molid633011', 'a/11808', 'a/15905']),
                          # (['IP1_0000', 'IP1_0001']),
                          # (['AP1_0100', 'IP1_0000']),
                          # (['HP1_0100', 'WP1_0010']),
                          # (['XP1_0005', 'KP1_0023']),
                          # (['EP1_0120', 'LP1_0099']),
                          # (['MP1_0019', 'FP1_0083']),
                          ])
def test_aids(aids_graphs, dataframe_aids, graph_name_source, graph_name_target, gr_name_src, gr_name_trgt):
    graph_source, graph_target = [graph for graph in aids_graphs if graph.name in [graph_name_source, graph_name_target]]
    print(graph_source)
    print(graph_target)
    cst_cost_node = 1.1
    cst_cost_edge = 0.1
    ged = GED(EditCostAIDS(cst_cost_node, cst_cost_node,
                           cst_cost_edge, cst_cost_edge, 'dirac'))

    results = ged.compute_edit_distance(graph_source, graph_target)
    expected = dataframe_aids.loc[gr_name_src, gr_name_trgt]

    print(f'###### diff {results - expected}')
    assert results == expected






