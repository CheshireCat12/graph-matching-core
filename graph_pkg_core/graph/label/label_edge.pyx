cdef class LabelEdge(LabelBase):

    def __init__(self, valence):
        self.valence = valence

    cpdef tuple get_attributes(self):
        return (self.valence, )