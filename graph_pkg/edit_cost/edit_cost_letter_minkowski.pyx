cdef class EditCostLetterMinkowski(EditCostLetter):

    def __cinit__(self, degree):
        self.degree = degree

    cdef double _compute_cost_substitute_node(self, double x1, double y1, double x2, double y2) except? -1:
        return c_sqrt(c_pow(x1 - x2, self.degree) + c_pow(y1 - y2, self.degree))
