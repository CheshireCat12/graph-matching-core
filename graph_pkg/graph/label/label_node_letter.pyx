cdef class LabelNodeLetter(LabelBase):

    def __cinit__(self, int x, int y):
        self.x = x
        self.y = y

    cpdef tuple get_attributes(self):
        return self.x, self.y
