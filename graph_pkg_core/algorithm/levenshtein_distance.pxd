cdef class LevenshteinDistance:

    cdef:
        readonly double[:, ::1] distances

    cpdef double compute_string_edit_distance_cpd(self, str string_1, str string_2, double subst_cost, double ins_cost, double del_cost)

    cdef double compute_string_edit_distance(self, str string_1, str string_2, double subst_cost, double ins_cost, double del_cost)

    cpdef double compute_string_edit_distance_normalized_cpd(self, str string_1, str string_2)

    cdef double compute_string_edit_distance_normalized(self, str string_1, str string_2)

