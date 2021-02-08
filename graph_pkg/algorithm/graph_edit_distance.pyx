import numpy as np
cimport cython

from libc.math cimport abs as c_abs
from scipy.optimize import linear_sum_assignment

cdef class GED:

    def __cinit__(self, EditCost edit_cost):
        self.edit_cost = edit_cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _create_c_matrix(self):
        cdef:
            int i, j
            int n, m
            double cost

        n = len(self.graph_source)
        m = len(self.graph_target)

        self.C = np.zeros((n + m, n + m), dtype=np.float64)

        # Create substitute part
        for i in range(n):
            for j in range(m):
                cost = self.edit_cost.cost_substitute_node(self.graph_source.nodes[i],
                                                           self.graph_target.nodes[j])
                self.C[i][j] = cost

        # Create node deletion part
        self.C[0:n:1, m:n+m:1] = np.inf
        for i in range(n):
            j = m + i
            cost = self.edit_cost.cost_delete_node(self.graph_source.nodes[i])

            self.C[i][j] = cost

        # Create node insertion part
        self.C[n:n+m:1, 0:m:1] = np.inf
        for j in range(m):
            i = n + j
            cost = self.edit_cost.cost_insert_node(self.graph_target.nodes[j])

            self.C[i][j] = cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _create_c_star_matrix(self):
        cdef:
            int i, j
            int n, m
            double cost

            int[::1] out_degrees_source, out_degrees_target

        n = len(self.graph_source)
        m = len(self.graph_target)

        out_degrees_source = self.graph_source.out_degrees()
        out_degrees_target = self.graph_target.out_degrees()

        self.C_star = np.zeros((n + m, n + m), dtype=np.float64)
        self.C_star[::1, ::1] = self.C

        # Update the substitution part
        for i in range(n):
            for j in range(m):
                cost = c_abs(out_degrees_source[i] - out_degrees_target[j])
                self.C_star[i][j] += cost

        # Update the deletion part
        for i in range(n):
            j = m + i
            self.C_star[i][j] += out_degrees_source[i]

        # Update the insertion part
        for j in range(m):
            i = n + j
            self.C_star[i][j] += out_degrees_target[j]



    cpdef double compute_distance_between_graph(self, Graph graph_source, Graph graph_target):
        cdef:
            int i, j
            int n, m

        self.graph_source = graph_source
        self.graph_target = graph_target

        self._create_c_matrix()
        self._create_c_star_matrix()

        row_ind, col_ind = linear_sum_assignment(self.C_star)

        print(row_ind)
        print(col_ind)
        print(self.C.base[row_ind, col_ind])
        # print(graph_source)
        # print(graph_target)

        n = len(self.graph_source)
        m = len(self.graph_target)

        for i in range(n + m):
            for j in range(i + 1, n + m):
                phi_i = col_ind[i]
                phi_j = col_ind[j]

                print(f'i {i}, j {j}')

                if graph_source.has_edge(i, j):
                    # check for edge substitution
                    if graph_target.has_edge(phi_i, phi_j):
                        print(f'-Exchange edge {(i, j)} --> {(phi_i, phi_j)}')

                    #check for edge insertion
                    else:
                        print(f'#insertion edge {(i, j)} --> {(phi_i, phi_j)}')
                else:
                    # check for edge deletion
                    if graph_target.has_edge(phi_i, phi_j):
                        print(f'*Deletion edge {(i, j)} --> {(phi_i, phi_j)}')



                # if i < n and  j >= m:
                #     if 0 <= phi_i <= n or 0 <= phi_j <= m:
                #         print(f'Exchange edge {(i, j)} --> {(phi_i, phi_j)}')

