from cyged.graph_pkg_core.graph.edge cimport Edge
from cyged.graph_pkg_core.graph.node cimport Node


cdef class Graph:
    cdef:
        readonly str name
        readonly str filename
        readonly list nodes
        readonly dict edges

        readonly unsigned int num_nodes_max
        public unsigned int num_nodes_current
        public int num_edges

        public int[:, ::1] adjacency_matrix

    cdef void _init_edges(self)

    cdef void _init_adjacency_matrix(self)

    cdef bint _does_node_exist(self, int idx_node)

    cpdef bint has_edge(self, int idx_start, int idx_end)

    cpdef list get_nodes(self)

    cpdef dict get_edges(self)

    cdef Edge get_edge_by_node_idx(self, int idx_node_start, int idx_node_end)

    cpdef int add_node(self, Node node) except? -1

    cpdef int add_edge(self, Edge edge) except? -1

    cpdef int[::1] degrees(self)

    cpdef void remove_node_by_idx(self, unsigned int idx_node)

    cpdef void remove_all_edges_by_node_idx(self, int idx_node)

    cdef void __del_edge(self, unsigned int idx_node, list edges)
