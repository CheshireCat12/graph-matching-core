import random


cdef class LabelNodeEmbedding(LabelBase):

    def __init__(self, double[::1] vector):
        self.vector = vector

    cpdef tuple get_attributes(self):
        return (self.vector, )

    def sigma_attributes(self):
        return f'Element: {self.vector.base}'

    def sigma_position(self):
        return [random.uniform(0, 1) for _ in range(2)]
