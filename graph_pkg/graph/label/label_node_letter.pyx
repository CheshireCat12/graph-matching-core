cdef class LabelNodeLetter(LabelBase):

    def __init__(self, double x, double y):
        self.x = x
        self.y = y

    cpdef tuple get_attributes(self):
        return self.x, self.y
