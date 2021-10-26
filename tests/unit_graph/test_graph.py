# -*- coding: utf-8 -*-
import pytest
from graph_pkg.graph.graph import Graph as gr
# from graph_gnn_embedding.graph.network import Network as Graph
from graph_pkg.graph.network import Node, Edge
from itertools import combinations
import numpy as np



#
# @pytest.fixture()
# def my_graph():
#     return Graph('new graph')
#
#
# @pytest.fixture()
# def constructed_graph(my_graph):
#     for i in range(0, 2):
#         my_graph.add_node(Node(i, f'data{i}'))
#
#     my_graph.add_edge(Edge(0, 1))
#
#     return my_graph
#
#
# @pytest.fixture()
# def clique(my_graph):
#     num_nodes = 4
#     for i in range(num_nodes):
#         my_graph.add_node(Node(i, f'data{i}'))
#
#     for n_idx1, n_idx2 in combinations(range(num_nodes), 2):
#         my_graph.add_edge(Edge(n_idx1, n_idx2))
#
#     return my_graph
#
#
# def test_label_of_graph(my_graph):
#     assert my_graph.get_name() == 'new graph'
#
#
# def test_insertion_node(my_graph):
#     tmp_node = Node(1, 'test_node')
#     my_graph.add_node(tmp_node)
#
#     nodes_expected = {1: tmp_node}
#
#     assert my_graph.get_nodes() == nodes_expected
#
#
# def test_insertion_multi_nodes(my_graph):
#     tmp_nodes = dict()
#     for i in range(3):
#         id_ = i
#         node_tmp = Node(id_)
#         tmp_nodes[id_] = node_tmp
#         my_graph.add_node(node_tmp)
#
#     assert my_graph.get_nodes() == tmp_nodes
#
#
# def test_insertion_edge(my_graph):
#     my_graph.add_node(Node(1, 'node1'))
#     my_graph.add_node(Node(2, 'node2'))
#
#     tmp_edge = Edge(1, 2)
#     reversed_tmp_edge = Edge(2, 1)
#     my_graph.add_edge(tmp_edge)
#
#     expected_edges_in = {1: [tmp_edge], 2: []}
#     expected_edges_out = {1: [], 2: [reversed_tmp_edge]}
#
#     assert my_graph.get_edges_in() == expected_edges_in
#     assert my_graph.get_edges_out() == expected_edges_out
#
#
# @pytest.mark.parametrize('number_of_nodes', [3, 10])
# def test_insertion_multiple_edges(my_graph, number_of_nodes):
#     for i in range(number_of_nodes):
#         my_graph.add_node(Node(i, f'node{i}'))
#
#     expected_edges_in = {i: [] for i in range(number_of_nodes)}
#     expected_edges_out = {i: [] for i in range(number_of_nodes)}
#
#     for e_idx1, e_idx2 in [(0, 1), (0, 2)]:
#         tmp_edge = Edge(e_idx1, e_idx2)
#         reversed_tmp_edge = Edge(e_idx2, e_idx1)
#         my_graph.add_edge(tmp_edge)
#
#         expected_edges_in[e_idx1].append(tmp_edge)
#         expected_edges_out[e_idx2].append(reversed_tmp_edge)
#
#     assert my_graph.get_edges_in() == expected_edges_in
#     assert my_graph.get_edges_out() == expected_edges_out
#
#
# @pytest.mark.parametrize('n_idx1, n_idx2, e_idx1, e_idx2, expected_error_msg',
#                          [(3, 4, 1, 2, 'The starting node 1 does not exist!'),
#                           (1, 3, 1, 2, 'The ending node 2 does not exist!'),
#                           (3, 1, 2, 1, 'The starting node 2 does not exist!')])
# def test_insertion_edge_not_existent_node(my_graph, n_idx1, n_idx2, e_idx1, e_idx2, expected_error_msg):
#     my_graph.add_node(Node(n_idx1))
#     my_graph.add_node(Node(n_idx2))
#
#     with pytest.raises(AssertionError) as execinfo:
#         my_graph.add_edge(Edge(e_idx1, e_idx2))
#
#     exception_msg = execinfo.value.args[0]
#     assert exception_msg == expected_error_msg
#
#
# def test_deletion_edge(constructed_graph):
#     constructed_graph.remove_edge_by_id(0, 1)
#
#     assert constructed_graph.get_edges_in() == {0: [], 1: []}
#     assert constructed_graph.get_edges_out() == {0: [], 1: []}
#
#
# def test_deletion_node(constructed_graph):
#     constructed_graph.remove_node_by_id(1)
#
#     assert len(constructed_graph.get_nodes()) == 1
#     assert constructed_graph.get_edges_in() == {0: []}
#     assert constructed_graph.get_edges_out() == {0: []}
#
#
# def test_deletion_edges_in_clique(clique):
#     clique.remove_node_by_id(1)
#     num_nodes = 4
#
#     expected_nodes, expected_edges_in, expected_edges_out = {}, {}, {}
#
#     for i in range(num_nodes):
#         if i == 1:
#             continue
#         expected_nodes[i] = Node(i, f'data{i}')
#
#     expected_edges_in = {
#         0: [],
#         2: [],
#         3: []
#     }
#
#     # TODO: see if I can maintain the edges like that also easier for the
#     #       adjacency matrix.
#     expected_edges_out = {
#         0: [Edge(0, 2), Edge(0, 3)],
#         2: [Edge(2, 0), Edge(2, 3)],
#         3: [Edge(3, 0), Edge(3, 2)]
#     }
#
#     # for n_idx1, n_idx2 in combinations(range(num_nodes), 2):
#     #     if n_idx1 == 1 or n_idx2 == 1:
#     #         continue
#     #
#     #     expected_edges_in[n_idx1].append(Edge(n_idx1, n_idx2))
#     #     expected_edges_out[n_idx2].append(Edge(n_idx2, n_idx1))
#
#     assert len(clique.get_nodes()) == 3
#     assert clique.get_nodes() == expected_nodes
#     # assert clique.get_edges_in() == expected_edges_in
#     assert clique.get_edges_out() == expected_edges_out
#
#
# def test_adjacency_matrix(constructed_graph):
#     assert np.array_equal(constructed_graph.adjacency_matrix(),
#                           np.array([[0, 1], [1, 0]]))
