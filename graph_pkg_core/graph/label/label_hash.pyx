import numpy as np
cimport numpy as np
from cpython.object cimport Py_EQ

cdef class LabelHash(LabelBase):
    """
    LabelNodeHash contains a list of hashes as attributes
    """
    def __init__(self, list hashes):
        """

        Args:
            hash: list
        """
        self.hashes = hashes

    cpdef tuple get_attributes(self):
        """

        Returns: Tuple(list, ) the array attribute

        """
        return (self.hashes,)

    def __richcmp__(self, LabelBase other, int op):
        assert isinstance(other, LabelHash), f'The element {str(other)} is not an Label!'
        cdef:
            str other_attr

        if op == Py_EQ:
            other_attr, *_ = other.get_attributes()

            return self.hashes == other_attr
        else:
            assert False
