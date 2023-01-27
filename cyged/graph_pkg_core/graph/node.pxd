from cyged.graph_pkg_core.graph.label.label_base cimport LabelBase


cdef class Node:
    cdef:
        readonly unsigned int idx
        readonly LabelBase label

    cdef void update_idx(self, unsigned int new_idx)