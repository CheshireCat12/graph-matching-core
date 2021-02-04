from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelNodeLetter(LabelBase):
    cdef:
        float x
        float y

    cpdef tuple get_attributes(self)