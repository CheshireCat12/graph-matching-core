import random


cdef class LabelNodeMutagenicity(LabelBase):

    def __init__(self, str chem):
        self.chem = chem
        self.chem_int = int.from_bytes(chem.encode(), 'little')

    cpdef tuple get_attributes(self):
        return (self.chem, )

    def sigma_attributes(self):
        return f'Element: {self.chem}'

    def sigma_position(self):
        return [random.uniform(0, 1) for _ in range(2)]
