cdef class EditCostLetterMinkowski(EditCostLetter):

    def __cinit__(self, degree):
        self.degree = degree

    cdef float _compute_cost_substitute_node(self, float x1, float y1, float x2, float y2) except? -1:
        return c_sqrt(c_pow(x1 - x2, self.degree) + c_pow(y1 - y2, self.degree))
