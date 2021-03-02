from itertools import combinations

import numpy as np
import pytest

from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.node import Node


@pytest.fixture()
def my_graph():
    return Graph('gr1', 'gr1.gxl', 2)

def test_simple_graph():
    my_graph = Graph('gr1', 'gr1.gxl', 1)

    assert my_graph.name == 'gr1'
    assert len(my_graph) == 0

# @pytest.mark.skip()
@pytest.mark.parametrize('num_nodes, idx_to_remove, expected_adj',
                         [(1, 0, np.array([[0]]))])
def test_remove_node(num_nodes, idx_to_remove, expected_adj):
    my_graph = Graph(f'gr{num_nodes}', f'gr{num_nodes}.gxl', num_nodes)
    nodes = []

    for i in range(num_nodes):
        tmp_node = Node(i, LabelNodeLetter(1+i, 1))
        nodes.append(tmp_node)
        my_graph.add_node(tmp_node)

    for idx_start, idx_end in combinations(range(num_nodes), 2):
        tmp_edge = Edge(idx_start, idx_end, LabelEdge(0))
        my_graph.add_edge(tmp_edge)

    my_graph.remove_node_by_idx(idx_to_remove)
    # nodes.pop(0)

    # expected edges
    # expected_adjacency_mat = np.array([[0, 1, 1],
    #                                    [1, 0, 1],
    #                                    [1, 1, 0]])

    print(f'--{nodes}')
    print(my_graph.get_nodes())
    # assert my_graph.get_nodes() == nodes
    assert len(my_graph) == num_nodes - 1
    # assert np.array_equiv(expected_adjacency_mat, my_graph.adjacency_matrix)
    # assert False

# TODO: remove other idx_node()
# TODO: delete_mutliple_nodes()

def test_copy_graph(my_graph):
    my_graph.add_node(Node(0, LabelNodeLetter(1, 1)))
    my_graph.add_node(Node(1, LabelNodeLetter(2, 3)))

    my_graph.add_edge(Edge(0, 1, LabelEdge(0)))

    import copy
    new_graph = copy.deepcopy(my_graph)
    my_graph.remove_node_by_idx(0)
    print(new_graph)
    print(my_graph)
    # assert False

@pytest.mark.parametrize('num_nodes',
                         [1, 5, 10])
def test_add_node(num_nodes):
    my_graph = Graph(f'gr{num_nodes}', f'gr{num_nodes}.gxl', num_nodes)
    nodes = []

    for i in range(num_nodes):
        tmp_node = Node(i, LabelNodeLetter(1, 1))
        nodes.append(tmp_node)
        my_graph.add_node(tmp_node)

    assert my_graph.get_nodes() == nodes
    assert len(my_graph) == num_nodes


@pytest.mark.parametrize('num_nodes, error_idx',
                         [(5, 5),
                          (5, 8),])
def test_add_node_higher_than_num_nodes(num_nodes, error_idx):

    my_graph = Graph(f'gr{num_nodes}', f'gr{num_nodes}.gxl', num_nodes)
    tmp_node = Node(error_idx, LabelNodeLetter(1, 1))

    with pytest.raises(AssertionError) as execinfo:
        my_graph.add_node(tmp_node)

    error_msg = execinfo.value.args[0]
    expected_error_msg = f'The idx of the node {error_idx} exceed the number of nodes {num_nodes} authorized!'
    assert error_msg == expected_error_msg


@pytest.mark.parametrize('num_nodes, expected_edges',
                         [(2, {0: [None, Edge(0, 1, LabelEdge(0))],
                               1: [Edge(1, 0, LabelEdge(0)), None]}),
                          (3, {0: [None, Edge(0, 1, LabelEdge(0)), Edge(0, 2, LabelEdge(0))],
                               1: [Edge(1, 0, LabelEdge(0)), None, Edge(1, 2, LabelEdge(0))],
                               2: [Edge(2, 0, LabelEdge(0)), Edge(2, 1, LabelEdge(0)), None]})
                          ])
def test_add_clique_edge(num_nodes, expected_edges):
    my_graph = Graph(f'gr{num_nodes}', f'gr{num_nodes}.gxl', num_nodes)

    for i in range(num_nodes):
        tmp_node = Node(i, LabelNodeLetter(i, i))
        my_graph.add_node(tmp_node)

    for idx_start, idx_end in combinations(range(num_nodes), 2):
        tmp_edge = Edge(idx_start, idx_end, LabelEdge(0))
        my_graph.add_edge(tmp_edge)

    assert my_graph.get_edges() == expected_edges
    assert my_graph.has_edge(0, num_nodes - 1) == True
    assert my_graph.has_edge(num_nodes - 1, 0) == True
    assert my_graph.has_edge(0, num_nodes + 1) == False


@pytest.mark.parametrize('num_nodes, nodes, edge, expected_error_msg',
                         [(5, [], Edge(7, 1, LabelEdge(0)), 'The starting node 7 does not exist!'),
                          (5, [Node(0, LabelNodeLetter(1, 1))], Edge(0, 23, LabelEdge(0)), 'The ending node 23 does not exist!')
                          ])
def test_insert_invalid_edge(num_nodes, nodes, edge, expected_error_msg):
    my_graph = Graph(f'gr{num_nodes}', f'gr{num_nodes}.gxl', num_nodes)

    for node in nodes:
        my_graph.add_node(node)

    with pytest.raises(AssertionError) as execinfo:
        my_graph.add_edge(edge)

    error_msg = execinfo.value.args[0]

    assert error_msg == expected_error_msg

@pytest.mark.parametrize('num_nodes, expected_matrix',
                         [(2, np.array([[0, 1],
                                        [1, 0]], dtype=np.int32)),
                          (3, np.array([[0, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 0]], dtype=np.int32))
                          ])
def test_adjacency_matrix(num_nodes, expected_matrix):
    my_graph = Graph(f'gr{num_nodes}', f'gr{num_nodes}.gxl', num_nodes)

    for i in range(num_nodes):
        tmp_node = Node(i, LabelNodeLetter(i, i))
        my_graph.add_node(tmp_node)

    for idx_start, idx_end in combinations(range(num_nodes), 2):
        tmp_edge = Edge(idx_start, idx_end, LabelEdge(0))
        my_graph.add_edge(tmp_edge)

    # transform memoryview to np.array
    # my_graph.adjacency_matrix.base
    # np.asarray(my_graph.adjacency_matrix)

    assert np.array_equal(np.asarray(my_graph.adjacency_matrix), expected_matrix)


@pytest.mark.parametrize('num_nodes, expected_matrix',
                         [(2, np.array([1, 1], dtype=np.int32)),
                          (3, np.array([2, 2, 2], dtype=np.int32))
                          ])
def test_out_in_degrees(num_nodes, expected_matrix):
    my_graph = Graph(f'gr{num_nodes}', f'gr{num_nodes}.gxl', num_nodes)

    for i in range(num_nodes):
        tmp_node = Node(i, LabelNodeLetter(i, i))
        my_graph.add_node(tmp_node)

    for idx_start, idx_end in combinations(range(num_nodes), 2):
        tmp_edge = Edge(idx_start, idx_end, LabelEdge(0))
        my_graph.add_edge(tmp_edge)

    assert np.array_equal(np.asarray(my_graph.out_degrees()), expected_matrix)
    # assert np.array_equal(np.asarray(my_graph.in_degrees()), expected_matrix)
