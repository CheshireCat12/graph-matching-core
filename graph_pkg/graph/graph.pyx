import numpy as np

cdef class Graph:
    """A class that is used to work with nodes and edges of a graph"""

    def __init__(self, str name, int num_nodes):
        self.name = name
        self.num_nodes_max = num_nodes
        self.num_nodes_current = 0
        self.nodes = [None] * num_nodes
        self._init_edges()
        self._init_adjacency_matrix()

    cdef void _init_edges(self):
        self.edges = {i: [] for i in range(self.num_nodes_max)}

    cdef void _init_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros((self.num_nodes_max, self.num_nodes_max),
                                         dtype=np.int32)

    cdef bint _does_node_exist(self, int idx_node):
        return 0 <= idx_node < self.num_nodes_max and \
               self.nodes[idx_node] is not None

    cpdef list get_nodes(self):
        return self.nodes

    cpdef int add_node(self, Node node) except? -1:
        assert node.idx < self.num_nodes_max, \
            f'The idx of the node {node.idx} exceed the number of nodes {self.num_nodes_max} authorized!'
        assert not self._does_node_exist(node.idx), \
            f'The position {node.idx} is already used!'

        self.nodes[node.idx] = node
        self.num_nodes_current += 1

    cpdef dict get_edges(self):
        return self.edges

    cpdef int add_edge(self, Edge edge) except? -1:
        assert self._does_node_exist(edge.idx_node_start), f'The starting node {edge.idx_node_start} does not exist!'
        assert self._does_node_exist(edge.idx_node_end), f'The ending node {edge.idx_node_end} does not exist!'

        cdef Edge reversed_edge = edge.reversed()
        self.edges[edge.idx_node_start].append(edge)
        self.edges[reversed_edge.idx_node_start].append(reversed_edge)

        self.adjacency_matrix[edge.idx_node_start][edge.idx_node_end] = 1
        self.adjacency_matrix[reversed_edge.idx_node_start][reversed_edge.idx_node_end] = 1

    def _set_edge(self):
        edges_set = set()
        for key, edges_lst in self.edges.items():
            edges_set.update(edges_lst)

        return edges_set

    def __str__(self):
        eof = ",\n\t\t"
        return f'Graph: \n' \
               f'\tName: {self.name}\n' \
               f'\tNumber of nodes max: {self.num_nodes_max}\n' \
               f'\tNodes: \n \t\t{eof.join(str(node) for node in self.nodes)}\n' \
               f'\tEdges: \n \t\t{eof.join(str(edge) for edge in self._set_edge())}\n'

    def __repr__(self):
        return f'Graph: {self.name} -> ' \
               f'Nodes: {", ".join(str(node) for node in self.nodes)}, ' \
               f'Edges: {", ".join(str(edge) for edge in self.edges)}'

    def __len__(self):
        return self.num_nodes_current
