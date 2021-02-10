cdef class LabelNodeMutagenicity(LabelBase):

    def __cinit__(self, str chem):
        self.chem = chem
        self.chem_int = sum(ord(letter) for letter in chem)

    cpdef tuple get_attributes(self):
        return (self.chem, )
