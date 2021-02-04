from graph_pkg.graph.label.label_node.label_node cimport LabelNode


cdef class Node:
    cdef:
        readonly unsigned int idx
        readonly LabelNode label