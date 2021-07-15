from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelNodeIMDB(LabelBase):
    cdef:
        readonly int actor

    cpdef tuple get_attributes(self)
