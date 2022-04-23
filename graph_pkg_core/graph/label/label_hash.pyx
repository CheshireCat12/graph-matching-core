import numpy as np
cimport numpy as np
from cpython.object cimport Py_EQ

cdef class LabelHash(LabelBase):
    """
    LabelNodeVector contains hash as str as attributes
    """
    def __init__(self, str hash):
        """

        Args:
            hash: str
        """
        self.hash = hash

    cpdef tuple get_attributes(self):
        """

        Returns: Tuple(str, ) the array attribute

        """
        return (self.hash,)

    def __richcmp__(self, LabelBase other, int op):
        assert isinstance(other, LabelHash), f'The element {str(other)} is not an Label!'
        cdef:
            str other_attr

        if op == Py_EQ:
            other_attr, *_ = other.get_attributes()

            return self.hash == other_attr
        else:
            assert False
