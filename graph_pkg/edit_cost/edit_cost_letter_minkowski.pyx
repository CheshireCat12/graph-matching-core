from libc.math cimport pow as c_pow
from libc.math cimport abs as c_abs
from libc.math cimport sqrt as c_sqrt

cdef class EditCostLetterMinkowski(EditCostLetter):

    def __cinit__(self, degree):
        self.degree = degree

    cdef double _compute_cost_substitute_node(self, double x1, double y1, double x2, double y2) except? -1:
        return c_abs(c_pow(c_pow(x1 - x2, self.degree) + c_pow(y1 - y2, self.degree), 1 / self.degree))
