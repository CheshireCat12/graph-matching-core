import random


cdef class LabelNodeIMDB(LabelBase):

    def __init__(self, int actor):
        self.actor = actor

    cpdef tuple get_attributes(self):
        return (self.actor, )

    def sigma_attributes(self):
        return f'Element: {self.actor}'

    def sigma_position(self):
        return [random.uniform(0, 1) for _ in range(2)]
