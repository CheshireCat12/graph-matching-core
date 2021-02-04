from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelEdge(LabelBase):

    cdef:
        readonly int valence

    cpdef tuple get_attributes(self)