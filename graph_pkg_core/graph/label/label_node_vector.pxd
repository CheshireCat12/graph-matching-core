from graph_pkg_core.graph.label.label_base cimport LabelBase
import numpy as np
cimport numpy as np

cdef class LabelNodeVector(LabelBase):
    cdef:
        readonly np.ndarray vector

    cpdef tuple get_attributes(self)
