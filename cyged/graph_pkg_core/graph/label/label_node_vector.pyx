import numpy as np
cimport numpy as np
from cpython.object cimport Py_EQ
from cyged.graph_pkg_core.graph.label.label_base cimport LabelBase


cdef class LabelNodeVector(LabelBase):
    """
    LabelNodeVector contains as an np.array as attributes
    """
    def __init__(self, np.ndarray vector):
        """

        Args:
            vector: ndarray
        """
        self.vector = vector

    cpdef tuple get_attributes(self):
        """
        
        Returns: Tuple(np.array, ) the array attribute

        """
        return (self.vector,)

    def __richcmp__(self, LabelBase other, int op):
        assert isinstance(other, LabelNodeVector), f'The element {str(other)} is not an Label!'
        cdef:
            np.ndarray other_attr

        if op == Py_EQ:
            other_attr, *_ = other.get_attributes()

            return np.array_equal(self.vector, other_attr)
        else:
            assert False
