from graph_pkg.graph.label.label_base cimport LabelBase


cdef class Edge:

    cdef:
        unsigned int idx
        readonly unsigned int idx_node_start
        readonly unsigned int idx_node_end
        readonly LabelBase weight