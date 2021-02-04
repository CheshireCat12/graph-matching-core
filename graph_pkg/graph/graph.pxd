from graph_pkg.graph.node cimport Node
from graph_pkg.graph.edge cimport Edge

cdef class Graph:
    cdef:
        readonly str name
        list nodes
        dict edges

        unsigned int num_nodes
        unsigned int num_edges

    # cdef bint _does_node_exist(self, int idx_node)
    #
    # cdef void _del_edge(self, int idx_node, list edges)

    cpdef list get_nodes(self)

    # cpdef list get_edges(self)

    cpdef int add_node(self, Node node) except? -1

    # cpdef void add_edge(self, Edge edge)
    #
    # cpdef void add_edge(self, int idx_node_start, int idx_node_end)
    #
    # cpdef void remove_node_by_idx(self, int idx_node)
    #
    # cpdef void remove_edge_by_node_idx(self, int idx_node_start, int idx_node_end)
    #
    # cpdef void remove_all_edges_by_node_idx(self, int idx_node)
