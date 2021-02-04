cdef class LabelNodeAIDS(LabelBase):

    def __cinit__(self, str symbol, int chem, int charge, float x, float y):
        self.symbol = symbol
        self.chem = chem
        self.charge = charge
        self.x = x
        self.y = y

