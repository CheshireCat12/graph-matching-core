
cdef double compute_edit_distance(str string_1, str string_2, double subst_cost, double ins_cost, double del_cost )

cdef double[:, ::1] _compute_edit_distance_matr(str string_1, str string_2, double subst_cost, double ins_cost, double del_cost)