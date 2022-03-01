import numpy as np
cimport numpy as np
cimport cython


cdef class LevenshteinDistance:
    cpdef double compute_string_edit_distance_cpd(self,str string_1, str string_2, double subst_cost, double ins_cost, double del_cost):
        return self.compute_string_edit_distance(string_1, string_2, subst_cost, ins_cost, del_cost)

    cdef double compute_string_edit_distance(self, str string_1, str string_2, double subst_cost, double ins_cost, double del_cost):
        cdef:
            int len_1, len_2
            list str_1_list, str_2_list

        len_1 = len(string_1)
        len_2 = len(string_2)
        str_1_list = list(string_1)
        str_2_list = list(string_2)

        # init empty distance array
        self.distances = np.zeros((len(string_1) + 1, len(string_2) + 1))


        for token_1 in range(len_1 + 1):
            for token_2 in range(len_2 + 1):
                if token_1 == 0:
                    self.distances[token_1][token_2] = token_2
                elif token_2 == 0:
                    self.distances[token_1][token_2] = token_1
                elif str_1_list[token_1 - 1] == str_2_list[token_2 - 1]:
                    # that works only under the assumption that if subst cost = 0, when the letters are the same
                    self.distances[token_1][token_2] = self.distances[token_1 -1][token_2 - 1]
                else:
                    dist_subst = self.distances[token_1 - 1][token_2 - 1] + subst_cost
                    dist_ins = self.distances[token_1][token_2 -1] + ins_cost
                    dist_del = self.distances[token_1-1][token_2] + del_cost
                    self.distances[token_1][token_2] = min(dist_ins,dist_del, dist_subst)


        return self.distances[len_1][len_2]



    cpdef double compute_string_edit_distance_normalized_cpd(self,str string_1, str string_2):
        return self.compute_string_edit_distance_normalized(string_1, string_2)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double compute_string_edit_distance_normalized(self, str string_1, str string_2):
        '''
        The normalized distance consists of the following formula:
        (1- ( (2* |lcs(x,y)|) / (|x| + |y|))) *2
        where the lcs is simply calculating the unnormalized string edit distance d(string_1, string_2) with costs of
        3,1,1 and then calculating (len(self.sequence_source) + len(self.sequence_target) - d(string_1, string_2) / 2
        :param string_1:
        :param string_2:
        :param subst_cost:
        :param ins_cost:
        :param del_cost:
        :return:
        '''
        cdef:
            int len_1, len_2
            list str_1_list, str_2_list

        len_1 = len(string_1)
        len_2 = len(string_2)
        lcs = (len_1+ len_2 - self.compute_string_edit_distance(string_1, string_2,2,1,1)) / 2

        return 2*(1- ((2*lcs) / (len_1+ len_2)))

