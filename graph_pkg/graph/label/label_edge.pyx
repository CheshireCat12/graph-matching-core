# cdef class LabelWeightedEdge(LabelBase):
#
#     def __cinit__(self, valence):
#         self.valence = valence

cdef class LabelEdge(LabelBase):

    # def __cinit__(self):
    #     self.valence = 0

    def __cinit__(self, valence):
        self.valence = valence

    cpdef tuple get_attributes(self):
        return (self.valence, )