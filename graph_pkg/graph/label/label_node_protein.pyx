cdef class LabelNodeProtein(LabelBase):

    def __init__(self, int type_, int aa_length, str sequence):
        self.type_ = type_
        self.aa_length = aa_length
        self.sequence = sequence

    cpdef tuple get_attributes(self):
        return self.type_, self.aa_length, self.sequence

    def sigma_attributes(self):
        raise NotImplementedError("Implement me pelase")

    def sigma_position(self):
        raise NotImplementedError("Implement me pelase")
