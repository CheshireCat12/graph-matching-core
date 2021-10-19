from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelNodeEmbedding(LabelBase):
    cdef:
        readonly double[::1] vector

    cpdef tuple get_attributes(self)
