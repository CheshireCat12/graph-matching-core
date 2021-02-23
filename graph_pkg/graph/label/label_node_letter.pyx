cdef class LabelNodeLetter(LabelBase):

    def __init__(self, double x, double y):
        self.x = x
        self.y = y

    cpdef tuple get_attributes(self):
        return self.x, self.y

    def json_attributes(self):
        return f'"x":{self.x}, "y":{self.y}'

