from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelNodeLetter(LabelBase):
    cdef:
        double x
        double y

    cpdef tuple get_attributes(self)