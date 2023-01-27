from cyged.graph_pkg_core.graph.label.label_base cimport LabelBase

cdef class LabelEdge(LabelBase):
    cdef:
        readonly int weight

    cpdef tuple get_attributes(self)
