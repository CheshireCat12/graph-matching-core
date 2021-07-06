import random


cdef class LabelNodeCollab(LabelBase):

    def __init__(self, int chem):
        self.chem = chem

    cpdef tuple get_attributes(self):
        return (self.chem, )

    def sigma_attributes(self):
        return f'Element: {self.chem}'

    def sigma_position(self):
        return [random.uniform(0, 1) for _ in range(2)]
