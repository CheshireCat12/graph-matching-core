import pickle
import sys

import numpy as np
import pytest
import random

from graph_pkg.algorithm.graph_edit_distance import GED
from graph_pkg.edit_cost.edit_cost_AIDS import EditCostAIDS
from graph_pkg.edit_cost.edit_cost_letter import EditCostLetter
from graph_pkg.edit_cost.edit_cost_mutagenicity import EditCostMutagenicity
from graph_pkg.edit_cost.edit_cost_NCI1 import EditCostNCI1
from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.label.label_node_mutagenicity import LabelNodeMutagenicity
from graph_pkg.graph.node import Node
from graph_pkg.loader.loader_AIDS import LoaderAIDS
from graph_pkg.loader.loader_letter import LoaderLetter
from graph_pkg.loader.loader_mutagenicity import LoaderMutagenicity
from graph_pkg.loader.loader_NCI1 import LoaderNCI1


@pytest.fixture()
def letter_graphs():
    loader = LoaderLetter()
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
def NCI1_graphs():
    loader = LoaderNCI1()
    graphs = loader.load()
    return graphs

@pytest.fixture()
def dataframe_letter():
    with open('./results/goal/anthony_ged_dist_mat_alpha_node_cost0.9_edge_cost2.3.pkl', 'rb') as file:
        df = pickle.load(file)

    return df


@pytest.fixture()
def dataframe_aids():
    with open('./results/goal/anthony_ged_dist_mat_alpha_node_cost1.1_edge_cost0.1.pkl', 'rb') as file:
        df = pickle.load(file)

    return df


@pytest.fixture()
def dataframe_mutagenicity():
    with open('./results/goal/anthony_ged_dist_mat_alpha_node_cost11.0_edge_cost1.1.pkl', 'rb') as file:
        df = pickle.load(file)

    return df


@pytest.fixture()
def define_graphs():
    ged = GED(EditCostLetter(1., 1., 1., 1., 'manhattan'))

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


def test_ged_same_graph(define_graphs):
    ged, graph_source, graph_target = define_graphs

    cost = ged.compute_edit_distance(graph_source, graph_source)

    expected_cost = 0.

    assert cost == expected_cost


@pytest.mark.parametrize('graph_source_target, accuracy',
                         [
                          (['AP1_0000', 'AP1_0001'], 1e-6),
                          (['IP1_0000', 'IP1_0001'], 1e-6),
                          (['AP1_0100', 'IP1_0000'], 1e-6),
                          (['HP1_0100', 'WP1_0010'], 1e-6),
                          (['XP1_0005', 'KP1_0023'], 1e-6),
                          (['EP1_0120', 'LP1_0099'], 1e-6),
                          (['MP1_0019', 'FP1_0083'], 1e-6),
                          ])
def test_with_verified_data(letter_graphs, dataframe_letter, graph_source_target, accuracy):
    gr_name_src, gr_name_trgt = [name[0] + '/' + name for name in graph_source_target]
    graph_source, graph_target = [graph for graph in letter_graphs if graph.name in graph_source_target]

    # print(graph_source)
    # print(graph_target)
    cst_cost_node = 0.9
    cst_cost_edge = 2.3
    ged = GED(EditCostLetter(cst_cost_node, cst_cost_node,
                             cst_cost_edge, cst_cost_edge, 'euclidean'))

    results = ged.compute_edit_distance(graph_source, graph_target)
    expected = dataframe_letter.loc[gr_name_src, gr_name_trgt]
    np.set_printoptions(precision=2)
    print(np.asarray(ged.C))
    print(np.asarray(ged.C_star))

    print(f'res {results}')
    print(f'exp {expected}')
    print(f'###### diff {results - expected}')
    # assert results == expected
    assert (results - expected) < accuracy
    # assert False

@pytest.mark.parametrize('graph_name_source, graph_name_target, gr_name_src, gr_name_trgt',
                         [
                          #   (['molid600779', 'molid409962', 'i/10151', 'i/10084']),
                          # (['molid624151', 'molid633011', 'a/11808', 'a/15905']),
                          # (['molid633011', 'molid624151', 'a/15905', 'a/11808']),
                          # (['molid660165', 'molid645098', 'i/27249', 'a/21376']),
                          # (['molid645098', 'molid660165', 'a/21376', 'i/27249']),
                          (['molid698506', 'molid624151', 'a/41540', 'a/11808'])
                          ])
def test_aids(aids_graphs, dataframe_aids, graph_name_source, graph_name_target, gr_name_src, gr_name_trgt):
    graph_source = [graph for graph in aids_graphs if graph.name == graph_name_source][0]
    graph_target = [graph for graph in aids_graphs if graph.name == graph_name_target][0]

    cst_cost_node = 1.1
    cst_cost_edge = 0.1
    ged = GED(EditCostAIDS(cst_cost_node, cst_cost_node,
                           cst_cost_edge, cst_cost_edge, 'dirac'))

    results = ged.compute_edit_distance(graph_source, graph_target)
    expected = dataframe_aids.loc[gr_name_src, gr_name_trgt]

    print(ged.C.base)
    print(ged.C_star.base)

    assert results == expected
    assert False

# @pytest.mark.skip()
# @pytest.mark.skip(reason='I have to had the expected accuracy')
# @pytest.mark.xfail(reason='I don\'t have the good value for mutagenicity to compare with')
@pytest.mark.parametrize('graph_name_source_target',
                         [
                          (['molecule_2767', 'molecule_2769']),
                          (['molecule_2769', 'molecule_2767']),
                          (['molecule_1897', 'molecule_1349']),
                          (['molecule_1897', 'molecule_1051']),
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
    # print(f'###### diff {results - expected}')
    # print(f'{graph_name_source_target}: new dist {results} - old dist {expected}')
    # print(f'exp {expected}')
    assert results == expected
    # assert False


@pytest.mark.parametrize('graph_name_source_target, expected',
                         [
                             (['molecule_2767', 'molecule_2769'], 383.9),
                             (['molecule_2769', 'molecule_2767'], 383.9),
                             (['molecule_1897', 'molecule_1349'], 133.1),
                             (['molecule_1897', 'molecule_1051'], 147.4),
                             # (['molecule_3726', 'molecule_3844'], 147.4),
                             # (['molecule_1108', 'molecule_1067'], 147.4),
                             # # (['molecule_3726', 'molecule_3844'], 147.4),

                         ])
def test_NCI1(NCI1_graphs, graph_name_source_target, expected):
    graph_name_source, graph_name_target = graph_name_source_target
    graph_source = [graph for graph in NCI1_graphs if graph.name == graph_name_source][0]
    graph_target = [graph for graph in NCI1_graphs if graph.name == graph_name_target][0]

    cst_cost_node = 1.0
    cst_cost_edge = 1.0
    ged = GED(EditCostNCI1(cst_cost_node, cst_cost_node,
                           cst_cost_edge, cst_cost_edge, 'dirac', alpha=0.9))

    results = ged.compute_edit_distance(graph_source, graph_target, heuristic=True)


    print(f'ged : {results}')

    assert round(results, 5) == expected


@pytest.mark.parametrize('graph_name_source_target',
                         [
                             (['molecule_2767', 'molecule_2769']),
                             (['molecule_2769', 'molecule_2767']),
                             (['molecule_1897', 'molecule_1349']),
                             (['molecule_1897', 'molecule_1051']),
                         ])
def test_heuristic_inverse(mutagenicity_graphs, graph_name_source_target):
    graph_name_source, graph_name_target = graph_name_source_target
    graph_source = [graph for graph in mutagenicity_graphs if graph.name == graph_name_source][0]
    graph_target = [graph for graph in mutagenicity_graphs if graph.name == graph_name_target][0]

    cst_cost_node = 11.0
    cst_cost_edge = 1.1
    ged = GED(EditCostMutagenicity(cst_cost_node, cst_cost_node,
                                   cst_cost_edge, cst_cost_edge, 'dirac'))

    results = ged.compute_edit_distance(graph_source, graph_target, heuristic=True)
    results_inv = ged.compute_edit_distance(graph_target, graph_source, heuristic=True)

    print(f'result: {results}')
    print(f'result_inv: {results_inv}')

    assert results == results_inv
    # assert False


@pytest.mark.parametrize('graph_name_source_target',
                         [
                             (['molecule_2767', 'molecule_2769']),
                             (['molecule_2769', 'molecule_2767']),
                             (['molecule_1897', 'molecule_1349']),
                             (['molecule_1897', 'molecule_1051']),
                         ])
def test_mutagenicity_with_deleted_nodes(mutagenicity_graphs, dataframe_mutagenicity, graph_name_source_target):
    gr_name_src, gr_name_trgt = ['mutagen/' + name for name in graph_name_source_target]
    graph_name_source, graph_name_target = graph_name_source_target
    graph_source = [graph for graph in mutagenicity_graphs if graph.name == graph_name_source][0]
    graph_target = [graph for graph in mutagenicity_graphs if graph.name == graph_name_target][0]

    cst_cost_node = 11.0
    cst_cost_edge = 1.1
    ged = GED(EditCostMutagenicity(cst_cost_node, cst_cost_node,
                                   cst_cost_edge, cst_cost_edge, 'dirac'))

    # Reproduce the source graph with more nodes
    new_gr_src = Graph(gr_name_src, 'gr.xls', len(graph_source)+2)
    for node in graph_source.nodes:
        new_gr_src.add_node(node)

    for idx, edges in graph_source.get_edges().items():
        for edge in edges:
            if edge is None:
                continue
            new_gr_src.add_edge(edge)

    new_gr_src.add_node(Node(len(graph_source), LabelNodeMutagenicity('C')))
    new_gr_src.add_node(Node(len(graph_source)+1, LabelNodeMutagenicity('N')))

    # Add random Edges
    for _ in range(4):
        new_gr_src.add_edge(Edge(len(graph_source),
                                 random.randint(0, len(graph_source)-1),
                                 LabelEdge(0)))

    for _ in range(6):
        new_gr_src.add_edge(Edge(len(graph_source) + 1,
                                 random.randint(0, len(graph_source)),
                                 LabelEdge(0)))




    new_gr_src.remove_node_by_idx(len(graph_source))
    new_gr_src.remove_node_by_idx(len(graph_source))

    results = ged.compute_edit_distance(new_gr_src, graph_target)
    expected = dataframe_mutagenicity.loc[gr_name_src, gr_name_trgt]
    # import numpy as np
    # np.savetxt(f'_c_{"X".join(graph_name_source_target)}.csv', np.asarray(ged.C), fmt='%10.3f', delimiter=';')
    # np.savetxt(f'c_star_{"X".join(graph_name_source_target)}.csv', np.asarray(ged.C_star), fmt='%10.3f', delimiter=';')
    print(f'###### diff {results - expected}')
    print(f'{graph_name_source_target}: new dist {results} - old dist {expected}')
    print(f'exp {expected}')
    assert results == expected
    # assert False
