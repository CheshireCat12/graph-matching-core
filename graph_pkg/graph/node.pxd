from cpython.object cimport Py_EQ

from graph_pkg.graph.label.label_base cimport LabelBase


cdef class Node:
    cdef:
        readonly unsigned int idx
        readonly LabelBase label