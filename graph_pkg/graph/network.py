from collections import defaultdict
from itertools import combinations
import numpy as np


class Node:

    def __init__(self, idx, data=None):
        self._idx = idx
        self._label = data

    def get_id(self):
        return self._idx

    def get_label(self):
        return self._label

    def __str__(self):
        return f'Node_id: {str(self._idx)}, '

    def __repr__(self):
        return f'Node_id: {str(self._idx)}, '

    def __eq__(self, other):
        assert isinstance(other, self.__class__), f'{other} is not a node.'
        return self._idx == other.get_id()

    def __hash__(self):
        return hash(self._idx)


class Edge:
    id_counter = 0

    def __init__(self, start_node, end_node, data=[]):
        assert start_node != end_node, 'No loops accepted, the nodes id must be different!'

        self.id = self.id_counter
        self.id_start_node = start_node
        self.id_end_node = end_node
        self.data = data

        self.id_counter += 1

    def __eq__(self, other):
        assert isinstance(other, self.__class__), f'{other} is not an edge.'
        return self.id_start_node == other.id_start_node and self.id_end_node == other.id_end_node

    def __hash__(self):
        return hash((self.id_start_node, self.id_end_node))

    def __str__(self):
        return f'{str(self.id_start_node)} is linked to {str(self.id_end_node)}\n'

    def __repr__(self):
        return f'{str(self.id_start_node)} is linked to {str(self.id_end_node)}\n'


class Network:

    def __init__(self, name=None):
        self._name = name
        self._nodes = dict()
        self._edges_in = dict()
        self._edges_out = dict()

        self._num_nodes = 0
        self._num_edges = 0

    def _does_node_exist(self, idx_node):
        return idx_node in self._nodes

    def get_name(self):
        return self._name

    def get_nodes(self):
        return self._nodes

    def get_edges_in(self):
        return self._edges_in

    def get_edges_out(self):
        return self._edges_out

    def get_edges(self):
        # TODO: merge correctly the edges in and out
        return self._edges_in + self._edges_out

    def add_node(self, node):
        self._nodes[node.get_id()] = node
        self._edges_in[node.get_id()] = list()
        self._edges_out[node.get_id()] = list()

        self._num_nodes += 1

    def add_edge(self, n_start, n_end):
        tmp_edge = Edge(n_start, n_end)

        self.add_edge(tmp_edge)

    def add_edge(self, edge):
        assert self._does_node_exist(edge.id_start_node), f'The starting node {edge.id_start_node} does not exist!'
        assert self._does_node_exist(edge.id_end_node), f'The ending node {edge.id_end_node} does not exist!'

        self._edges_in[edge.id_start_node].append(edge)
        reversed_edge = Edge(edge.id_end_node, edge.id_start_node)
        self._edges_out[reversed_edge.id_start_node].append(reversed_edge)

    def remove_node_by_id(self, n_idx):
        del self._nodes[n_idx]
        self.remove_all_edges_by_id(n_idx)

        self._num_nodes -= 1

    def __del_edge(self, n_idx, edges):
        for idx, element in enumerate(edges):
            if element.id_end_node == n_idx:
                edges.pop(idx)
                break

    def remove_all_edges_by_id(self, n_idx):
        edges_to_remove = self._edges_in[n_idx] + self._edges_out[n_idx]

        for edge in edges_to_remove:
            self.__del_edge(n_idx, self._edges_in[edge.id_end_node])
            self.__del_edge(n_idx, self._edges_out[edge.id_end_node])

        del self._edges_in[n_idx]
        del self._edges_out[n_idx]

    def remove_edge_by_id(self, node_start_idx, node_end_idx):
        assert self._does_node_exist(node_start_idx), 'The starting node id does not exist!'
        assert self._does_node_exist(node_end_idx), 'The ending node id does not exist!'

        self.__del_edge(node_end_idx, self._edges_in[node_start_idx])
        self.__del_edge(node_start_idx, self._edges_out[node_end_idx])

    def adjacency_matrix(self):
        adjacency = np.zeros((self._num_nodes, self._num_nodes))
        for idx, edge in self._edges_out.items():
            pass
            # adjacency[idx]


        return np.array([])

    def __str__(self):
        return f'Name: {self._name}\n'\
               f'Nodes: {" ".join(self._nodes)}\n' \
               f'Edges: \n ' \
               f'{" ".join(" ".join(str(edge) for edge in edges) for idx, edges in self._edges_in.items())}' \
               f'######\n' \
               f'{" ".join(" ".join(str(edge) for edge in edges) for idx, edges in self._edges_out.items())}'