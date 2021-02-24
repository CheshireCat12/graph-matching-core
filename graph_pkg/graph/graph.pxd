
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node


cdef class Graph:
    cdef:
        readonly str name
        readonly str filename
        list nodes
        dict edges

        readonly unsigned int num_nodes_max
        unsigned int num_nodes_current
        unsigned int num_edges

        readonly int[:, ::1] adjacency_matrix

    cdef void _init_edges(self)

    cdef void _init_adjacency_matrix(self)

    cdef bint _does_node_exist(self, int idx_node)

    cpdef bint has_edge(self, int idx_start, int idx_end)

    # cdef void _del_edge(self, int idx_node, list edges)

    cpdef list get_nodes(self)

    cpdef dict get_edges(self)

    cdef Edge get_edge_by_node_idx(self, int idx_node_start, int idx_node_end)

    cpdef int add_node(self, Node node) except? -1

    cpdef int add_edge(self, Edge edge) except? -1

    cpdef int[::1] in_degrees(self)

    cpdef int[::1] out_degrees(self)

    # cpdef void add_edge(self, int idx_node_start, int idx_node_end)
    #
    cpdef void remove_node_by_idx(self, int idx_node)
    #
    # cpdef void remove_edge_by_node_idx(self, int idx_node_start, int idx_node_end)

    cpdef void remove_all_edges_by_node_idx(self, int idx_node)

    cdef void __del_edge(self, int idx_node, list edges)
