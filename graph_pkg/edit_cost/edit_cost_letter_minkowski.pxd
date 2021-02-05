from libc.math cimport pow as c_pow, sqrt as c_sqrt

from graph_pkg.edit_cost.edit_cost_letter cimport EditCostLetter

cdef class EditCostLetterMinkowski(EditCostLetter):

    cdef:
        int degree

    cdef float _compute_cost_substitute_node(self, float x1, float y1, float x2, float y2) except? -1