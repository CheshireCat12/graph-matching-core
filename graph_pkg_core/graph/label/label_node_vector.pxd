from graph_pkg_core.graph.label.label_base cimport LabelBase
cimport numpy as cnp


cdef class LabelNodeVector(LabelBase):
    cdef:
        readonly cnp.ndarray vector

    cpdef tuple get_attributes(self)
