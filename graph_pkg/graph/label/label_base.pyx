cdef class LabelBase:

    def __cinit__(self):
        pass

    cpdef tuple get_attributes(self):
        """
        Take the heterogeneous attributes of the subclasses and return them.
        
        :return: tuple() of attributes 
        """
        raise NotImplementedError

    def __repr__(self):
        return f'Label attributes: {", ".join(str(element) for element in self.get_attributes())}'

