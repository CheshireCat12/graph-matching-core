import numpy

cpdef double compute_edit_distance_cpd(str string_1, str string_2, double subst_cost, double ins_cost, double del_cost):
    return compute_edit_distance(string_1, string_2, subst_cost, ins_cost, del_cost)

cdef double compute_edit_distance(str string_1, str string_2, double subst_cost, double ins_cost, double del_cost):
    len_1 = len(string_1)
    len_2 = len(string_2)
    return _compute_edit_distance_matr(string_1, string_2, subst_cost, ins_cost, del_cost)[len_1][len_2]

cpdef double[:, ::1] _compute_edit_distance_matr_cpd(str string_1, str string_2, double subst_cost, double ins_cost, double del_cost):
    return _compute_edit_distance_matr(string_1, string_2, subst_cost, ins_cost, del_cost)

cdef double[:, ::1] _compute_edit_distance_matr(str string_1, str string_2, double subst_cost, double ins_cost, double del_cost):
    cdef:
        int len_1, len_2
        list str_1_list, str_2_list
        double[:, ::1] distances

    len_1 = len(string_1)
    len_2 = len(string_2)
    str_1_list = list(string_1)
    str_2_list = list(string_2)

    # init empty distance array
    distances = numpy.zeros((len_1 + 1, len_2 + 1))

    for token_1 in range(len_1 + 1):
        for token_2 in range(len_2 + 1):
            if token_1 == 0:
                distances[token_1][token_2] = token_2
            elif token_2 == 0:
                distances[token_1][token_2] = token_1
            elif str_1_list[token_1 - 1] == str_2_list[token_2 - 1]:
                # that works only under the assumption that if subst cost = 0, when the letters are the same
                distances[token_1][token_2] = distances[token_1 -1][token_2 - 1]
            else:
                dist_subst = distances[token_1 - 1][token_2 - 1] + subst_cost
                dist_ins = distances[token_1 ][token_2 -1] + ins_cost
                dist_del = distances[token_1-1][token_2] + del_cost
                distances[token_1][token_2] = min(dist_ins,dist_del, dist_subst)

    return distances
