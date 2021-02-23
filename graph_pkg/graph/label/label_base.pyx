from cpython.object cimport Py_EQ


cdef class LabelBase:

    def __init__(self):
        pass

    cpdef tuple get_attributes(self):
        """
        Take the heterogeneous attributes of the subclasses and return them.
        
        :return: tuple() of attributes 
        """
        raise NotImplementedError

    def __richcmp__(self, LabelBase other, int op):
        assert isinstance(other, LabelBase), f'The element {str(other)} is not an Label!'
        cdef:
            tuple self_attr
            tuple other_attr

        if op == Py_EQ:
            self_attr = self.get_attributes()
            other_attr = other.get_attributes()

            return len(self_attr) == len(other_attr) and \
                   all(attr1 == attr2 for attr1, attr2 in zip(self_attr, other_attr))
        else:
            assert False

    def __repr__(self):
        return f'Label attributes: {", ".join(str(element) for element in self.get_attributes())}'

