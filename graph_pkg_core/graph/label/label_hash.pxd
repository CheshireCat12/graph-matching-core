from graph_pkg_core.graph.label.label_base cimport LabelBase
cimport numpy as cnp


cdef class LabelHash(LabelBase):
    cdef:
        readonly list hashes

    cpdef tuple get_attributes(self)
