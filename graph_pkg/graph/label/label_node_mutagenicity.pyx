cdef class LabelNodeMutagenicity(LabelBase):

    def __init__(self, str chem):
        self.chem = chem
        self.chem_int = int.from_bytes(chem.encode(), 'little')

    cpdef tuple get_attributes(self):
        return (self.chem, )
