from libc.math cimport pow as c_pow, sqrt as c_sqrt

from graph_pkg.edit_cost.edit_cost_letter cimport EditCostLetter

cdef class EditCostLetterMinkowski(EditCostLetter):

    cdef:
        int degree

    cdef double _compute_cost_substitute_node(self, double x1, double y1, double x2, double y2) except? -1