cdef class LabelNodeLetter(LabelBase):

    def __init__(self, double x, double y):
        self.x = x
        self.y = y

    cpdef tuple get_attributes(self):
        return self.x, self.y

    def sigma_attributes(self):
        return f'x:{self.x}, y:{self.y}'

    def sigma_position(self):
        return self.x, self.y

