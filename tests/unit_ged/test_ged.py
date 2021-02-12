import pickle
import sys

import networkx as nx
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


@pytest.fixture()
def letter_graphs():
    loader = LoaderLetter('HIGH')
    graphs = loader.load()
    return graphs


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
def dataframe_mutagenicity():
    with open('./data/goal/anthony_ged_dist_mat_alpha_node_cost11.0_edge_cost1.1.pkl', 'rb') as file:
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

def create_graphNX(graph):
    gr = nx.Graph()
    for idx, node in enumerate(graph.get_nodes()):
        if node is None:
            continue
        print(node.label)
        x, y = node.label.get_attributes()
        lbl = {'x': x, 'y': y}
        gr.add_node(idx, **lbl)

    for edge in graph._set_edge():
        if edge is None:
            continue
        gr.add_edge(edge.idx_node_start, edge.idx_node_end)

    return gr

@pytest.mark.skip()
@pytest.mark.parametrize('name_graphs_to_test',
                         [(['IP1_0000', 'IP1_0001']),
                          (['AP1_0000', 'AP1_0001']),
                          (['AP1_0100', 'IP1_0000']),
                          (['HP1_0100', 'WP1_0010']),
                          # (['XP1_0005', 'KP1_0023']),
                          # (['EP1_0120', 'LP1_0099']),
                          # (['MP1_0019', 'FP1_0083']),
                         ])
def test_letter_with_networkX(letter_graphs, name_graphs_to_test):
    print(name_graphs_to_test)
    graphs = [graph for graph in letter_graphs if graph.name in name_graphs_to_test]
    epsilon = 1e-14
    cst_cost_node = 0.9
    cst_cost_edge = 2.3
    ged = GED(EditCostLetter(cst_cost_node, cst_cost_node,
                             cst_cost_edge, cst_cost_edge, 'euclidean'))

    # gr1 = nx.Graph()
    # gr1.add_node(0, x=1.66831, y=2.93501)
    # gr1.add_node(1, x=1.34125, y=0.290566)
    # gr1.add_edge(0, 1)
    #
    # gr2 = nx.Graph()
    # gr2.add_node(0, x=1.18827, y=3.20702)

    gr1 = create_graphNX(graphs[0])
    gr2 = create_graphNX(graphs[1])

    from time import time
    start_time = time()
    expected_cost = nx.algorithms.graph_edit_distance(gr1, gr2,
                                                      node_subst_cost=lambda x, y: np.linalg.norm(np.array(list(x.values()))-np.array(list(y.values()))),
                                                      node_ins_cost=lambda x: cst_cost_node,
                                                      node_del_cost=lambda x: cst_cost_node,
                                                      # edge_subst_cost=lambda x, y: 0.,
                                                      edge_ins_cost=lambda x: cst_cost_edge,
                                                      edge_del_cost=lambda x: cst_cost_edge)
    expected_path = nx.algorithms.optimal_edit_paths(gr1, gr2,
                                                      node_subst_cost=lambda x, y: np.linalg.norm(np.array(list(x.values()))-np.array(list(y.values()))),
                                                      node_ins_cost=lambda x: cst_cost_node,
                                                      node_del_cost=lambda x: cst_cost_node,
                                                      # edge_subst_cost=lambda x, y: 0.,
                                                      edge_ins_cost=lambda x: cst_cost_edge,
                                                      edge_del_cost=lambda x: cst_cost_edge)


    print(f'Computation time NX {(time()-start_time) * 1000}')

    start_time = time()
    real_cost = ged.compute_edit_distance(graphs[0],
                                          graphs[1])
    print(f'Computation time {(time()-start_time) * 1000}')
    print(f'Expected cost: {expected_cost}')
    print(f'My cost:       {real_cost}')
    print(f'Path {expected_path[0][0][0]}')
    print(f'Path_edge {expected_path[0][0][1]}')
    print(f'Resd {ged.phi.base}')
    print(ged.C.base)
    print(ged.C_star)
    print('44444')
    assert abs(real_cost - expected_cost) < epsilon
    assert False
    # assert False

@pytest.mark.parametrize('graph_source_target, accuracy',
                         [(['AP1_0000', 'AP1_0001'], 1e-7),
                          (['IP1_0000', 'IP1_0001'], 1e-7),
                          (['AP1_0100', 'IP1_0000'], 1e-7),
                          (['HP1_0100', 'WP1_0010'], 1e-7),
                          (['XP1_0005', 'KP1_0023'], 1e-7),
                          (['EP1_0120', 'LP1_0099'], 1e-7),
                          (['MP1_0019', 'FP1_0083'], 1e-7),
                          ])
def test_with_verified_data(letter_graphs, dataframe_letter, graph_source_target, accuracy):
    gr_name_src, gr_name_trgt = [name[0] + '/' + name for name in graph_source_target]
    graph_source, graph_target = [graph for graph in letter_graphs if graph.name in graph_source_target]

    cst_cost_node = 0.9
    cst_cost_edge = 2.3
    ged = GED(EditCostLetter(cst_cost_node, cst_cost_node,
                             cst_cost_edge, cst_cost_edge, 'euclidean'))

    results = ged.compute_edit_distance(graph_source, graph_target)
    expected = dataframe_letter.loc[gr_name_src, gr_name_trgt]

    print(f'res {results}')
    print(f'exp {expected}')
    print(f'###### diff {results - expected}')
    # assert results == expected
    assert (results - expected) < accuracy

@pytest.mark.xfail(reason='I don\'t have the good value for AIDS to compare with')
@pytest.mark.parametrize('graph_name_source, graph_name_target, gr_name_src, gr_name_trgt',
                         [(['molid624151', 'molid633011', 'a/11808', 'a/15905']),
                          (['molid633011', 'molid624151', 'a/15905', 'a/11808']),
                          (['molid660165', 'molid645098', 'i/27249', 'a/21376']),
                          (['molid645098', 'molid660165', 'a/21376', 'i/27249']),
                          ])
def test_aids(aids_graphs, dataframe_aids, graph_name_source, graph_name_target, gr_name_src, gr_name_trgt):
    graph_source = [graph for graph in aids_graphs if graph.name == graph_name_source][0]
    graph_target = [graph for graph in aids_graphs if graph.name == graph_name_target][0]
    # print(graph_source)
    # print(graph_target)
    n, m = len(graph_source), len(graph_target)
    cst_cost_node = 1.1
    cst_cost_edge = 0.1
    ged = GED(EditCostAIDS(cst_cost_node, cst_cost_node,
                           cst_cost_edge, cst_cost_edge, 'dirac'))

    results = ged.compute_edit_distance(graph_source, graph_target)
    expected = dataframe_aids.loc[gr_name_src, gr_name_trgt]
    np.set_printoptions(precision=5, linewidth=1000, threshold=sys.maxsize)
    print(ged.C.base)
    print('@@@@@@@')
    # print(ged.C_star.base)
    # {gr_name_src}_{gr_name_trgt}
    # np.savetxt(f'_c_{gr_name_src.replace("/", "_")}_{gr_name_trgt.replace("/", "_")}.csv', np.asarray(ged.C), fmt='%10.3f', delimiter=';')
    # np.savetxt(f'_c_star_{gr_name_src.replace("/", "_")}_{gr_name_trgt.replace("/", "_")}.csv', np.asarray(ged.C_star), fmt='%10.3f', delimiter=';')
    # print('transpose')
    # test = np.asarray(ged.C)
    # print(np.transpose(test[:n, :m]))

    print(f'###### diff {results - expected}')
    print(f'res {results}')
    print(f'exp {expected}')
    assert results == expected
    assert False

# @pytest.mark.skip()
# @pytest.mark.skip(reason='I have to had the expected accuracy')
@pytest.mark.xfail(reason='I don\'t have the good value for mutagenicity to compare with')
@pytest.mark.parametrize('graph_name_source_target',
                         [(['molecule_2767', 'molecule_2769']),
                          (['molecule_2769', 'molecule_2767']),
                          ])
def test_mutagenicity(mutagenicity_graphs, dataframe_mutagenicity, graph_name_source_target):
    gr_name_src, gr_name_trgt = ['mutagen/' + name for name in graph_name_source_target]
    graph_name_source, graph_name_target = graph_name_source_target
    graph_source = [graph for graph in mutagenicity_graphs if graph.name == graph_name_source][0]
    graph_target = [graph for graph in mutagenicity_graphs if graph.name == graph_name_target][0]
    # print(graph_source)
    # print(graph_target)
    cst_cost_node = 11.0
    cst_cost_edge = 1.1
    ged = GED(EditCostMutagenicity(cst_cost_node, cst_cost_node,
                           cst_cost_edge, cst_cost_edge, 'dirac'))



    results = ged.compute_edit_distance(graph_source, graph_target)
    expected = dataframe_mutagenicity.loc[gr_name_src, gr_name_trgt]
    # import numpy as np
    # np.savetxt(f'_c_{"X".join(graph_name_source_target)}.csv', np.asarray(ged.C), fmt='%10.3f', delimiter=';')
    # np.savetxt(f'c_star_{"X".join(graph_name_source_target)}.csv', np.asarray(ged.C_star), fmt='%10.3f', delimiter=';')
    print(f'###### diff {results - expected}')
    print(f'{graph_name_source_target}: new dist {results} - old dist {expected}')
    print(f'exp {expected}')
    assert results == expected
    # assert False




