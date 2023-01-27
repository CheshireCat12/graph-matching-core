from cpython.object cimport Py_EQ

cdef class LabelBase:
    """
    Base class for the labels.
    The label classes contains the node attributes of the graph.

    Those attributes can be heterogeneous, so a specific application
    can inherit from this LabelBase class to work smoothly within this lib.
    """

    def __init__(self):
        pass

    cpdef tuple get_attributes(self):
        """
        Take the heterogeneous attributes of the subclasses and return them.
        
        Returns: tuple() of attributes 

        """
        raise NotImplementedError

    def __richcmp__(self, LabelBase other_lbl, int op):
        """
        Equal operator in cython for complex objects

        Args:
            other_lbl:
            op:

        Returns: Boolean - result of the comparison

        """

        assert isinstance(other_lbl, LabelBase), f'The element {str(other_lbl)} is not an Label!'
        cdef:
            tuple self_attr
            tuple other_attr

        if op == Py_EQ:
            self_attr = self.get_attributes()
            other_attr = other_lbl.get_attributes()

            return len(self_attr) == len(other_attr) and \
                   all(attr1 == attr2 for attr1, attr2 in zip(self_attr, other_attr))
        else:
            assert False

    def __repr__(self):
        return f'{", ".join(str(element) for element in self.get_attributes())}'

    def __str__(self):
        return f'Label attributes: {", ".join(str(element) for element in self.get_attributes())}'
