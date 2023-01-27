cimport cython
import numpy as np
cimport numpy as np

from cyged.graph_pkg_core.graph.edge cimport Edge
from cyged.graph_pkg_core.graph.node cimport Node


cdef class Graph:
    """
    A class that is used to work with nodes and edges of a graph

    Attributes
    ----------
    name : str
    filename : str
    nodes : list
    edges : dict
    num_nodes_max : int
    num_nodes_current : int
    num_edges : int
    adjacency_matrix : int[:, ::1]

    Methods
    -------

    """

    def __init__(self, str name, str filename, int num_nodes):
        """

        Args:
            name:
            filename:
            num_nodes:
        """
        self.name = name
        self.filename = filename
        self.num_nodes_max = num_nodes
        self.num_nodes_current = 0
        self.nodes = [None] * num_nodes
        self._init_edges()
        self._init_adjacency_matrix()


    cdef void _init_edges(self):
        """Init empty edges dict with corresponding number of nodes"""
        self.num_edges = 0
        self.edges = {i: [None] * self.num_nodes_max for i in range(self.num_nodes_max)}

    cdef void _init_adjacency_matrix(self):
        """Init empty adjacency matrix"""
        self.adjacency_matrix = np.zeros((self.num_nodes_max, self.num_nodes_max),
                                         dtype=np.int32)

    cdef bint _does_node_exist(self, int idx_node):
        """
        Check if the given node exist.
        
        The given node idx has to be 0 <= idx < |G|
        and the node has to be in the nodes list
        
        Args:
            idx_node: 

        Returns: bool

        """
        return 0 <= idx_node < int(self.num_nodes_max) and \
               self.nodes[idx_node] is not None

    cpdef bint has_edge(self, int idx_start, int idx_end):
        """
        Check if the edge exist by its starting and ending node idx
        
        Args:
            idx_start: 
            idx_end: 

        Returns:

        """
        return 0 <= idx_start < int(self.num_nodes_max) and \
               0 <= idx_end < int(self.num_nodes_max) and \
               self.edges[idx_start][idx_end] is not None

    cpdef list get_nodes(self):
        """
        
        Returns: list of nodes

        """
        return self.nodes

    cpdef int add_node(self, Node node) except? -1:
        """
        Add a node to the graph.
        
        Args:
            node: 

        Returns:

        """
        assert node.idx < self.num_nodes_max, \
            f'The idx of the node {node.idx} exceed the number of nodes {self.num_nodes_max} authorized!'
        assert not self._does_node_exist(node.idx), \
            f'The position {node.idx} is already used!'

        self.nodes[node.idx] = node
        self.num_nodes_current += 1

    cpdef dict get_edges(self):
        """
        
        Return: return the dict of edges
        
        """
        return self.edges

    cdef Edge get_edge_by_node_idx(self, int idx_node_start, int idx_node_end):
        """
        
        !! Caution, there is no verification if the indices exist!!
        
        Args:
            idx_node_start: 
            idx_node_end: 

        Returns: Edge corresponding to the given nodes_idx

        """
        return self.edges[idx_node_start][idx_node_end]

    cpdef int add_edge(self, Edge edge) except? -1:
        assert self._does_node_exist(edge.idx_node_start), \
            f'The starting node {edge.idx_node_start} does not exist!'
        assert self._does_node_exist(edge.idx_node_end), \
            f'The ending node {edge.idx_node_end} does not exist!'

        cdef Edge reversed_edge = edge.reversed()
        self.edges[edge.idx_node_start][edge.idx_node_end] = edge
        self.edges[reversed_edge.idx_node_start][reversed_edge.idx_node_end] = reversed_edge

        self.adjacency_matrix[edge.idx_node_start][edge.idx_node_end] = 1
        self.adjacency_matrix[reversed_edge.idx_node_start][reversed_edge.idx_node_end] = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int[::1] degrees(self):
        """
        Compute the degree of each node of the graph
        
        Returns: Memoryview of degrees

        """
        cdef:
            int idx
            int[::1] degrees

        size_adj = self.adjacency_matrix.shape[0]
        degrees = np.zeros(size_adj, dtype=np.int32)
        for idx, row in enumerate(self.adjacency_matrix):
            degrees[idx] += np.sum(row)

        return degrees

    cpdef void remove_node_by_idx(self, unsigned int idx_node):
        """
        Remove the node by the given index.
        
        Args:
            idx_node: 

        Returns:

        """
        assert self._does_node_exist(idx_node), f'The node {idx_node} can\'t be deleted'

        cdef Node node

        del self.nodes[idx_node]
        for node in self.nodes:
            if node is None:
                continue
            if node.idx > idx_node:
                node.update_idx(node.idx-1)
        self.num_nodes_current -= 1

        # Reduce the number max of nodes authorized
        # It is not a bug, it is a feature!
        self.num_nodes_max -= 1

        self.remove_all_edges_by_node_idx(idx_node)

    cpdef void remove_all_edges_by_node_idx(self, int idx_node):
        """
        Remove all the edges containing the given node index.
        
        :param idx_node: 
        :return: 
        """
        del self.edges[idx_node]
        tmp_edges = {}
        for key, edges in self.edges.items():
            self.__del_edge(idx_node, edges)

            if key > idx_node:
                key -= 1
            tmp_edges[key] = edges

        self.edges = tmp_edges
        self.adjacency_matrix = np.delete(self.adjacency_matrix, idx_node, 0)
        self.adjacency_matrix = np.delete(self.adjacency_matrix, idx_node, 1)

    cdef void __del_edge(self, unsigned int idx_node, list edges):
        cdef:
            Edge edge
            int idx_to_pop= -1
            int idx

        for idx, edge in enumerate(edges):
            if edge is None:
                continue
            if edge.idx_node_end == idx_node:
                idx_to_pop = idx
            if edge.idx_node_start > idx_node:
                edge.update_idx_node_start(edge.idx_node_start-1)
            if edge.idx_node_end > idx_node:
                edge.update_idx_node_end(edge.idx_node_end-1)

        if idx_to_pop >= 0:
            edges.pop(idx_to_pop)


    def _set_edge(self):
        edges_set = set()
        for key, edges_lst in self.edges.items():
            edges_set.update(edges_lst)

        edges_set.remove(None)
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

    def __reduce__(self):
        """Define how instance of Graph are pickled."""
        d = dict()
        d['name'] = self.name
        d['filename'] = self.filename
        d['nodes'] = self.nodes
        d['edges'] = self.edges
        d['num_nodes_max'] = self.num_nodes_max
        d['num_nodes_current'] = self.num_nodes_current
        d['num_edges'] = self.num_edges
        d['adjacency_matrix'] = np.asarray(self.adjacency_matrix)
        return (rebuild, (d,))


def rebuild(data):
    cdef:
        Graph g
    g = Graph(data['name'], data['filename'], data['num_nodes_max'])
    g.nodes = data['nodes']
    g.edges = data['edges']
    g.num_nodes_current = data['num_nodes_current']
    g.num_edges = data['num_edges']
    g.adjacency_matrix = data['adjacency_matrix']

    return g
