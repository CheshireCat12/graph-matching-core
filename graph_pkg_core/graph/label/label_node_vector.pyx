import random


cdef class LabelNodeVector(LabelBase):


    def __init__(self, cnp.ndarray vector):
        self.vector = vector

    cpdef tuple get_attributes(self):
        return (self.vector, )

    def sigma_attributes(self):
        return f'Element: {self.vector}'

    def sigma_position(self):
        return [random.uniform(0, 1) for _ in range(2)]
