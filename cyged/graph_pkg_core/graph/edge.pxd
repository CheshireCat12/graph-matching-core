from cyged.graph_pkg_core.graph.label.label_base cimport LabelBase


cdef class Edge:

    cdef:
        unsigned int idx
        readonly unsigned int idx_node_start
        readonly unsigned int idx_node_end
        readonly LabelBase weight

    cdef void update_idx_node_start(self, unsigned int new_idx_node_start)

    cdef void update_idx_node_end(self, unsigned int new_idx_node_end)

    cpdef Edge reversed(self)

