import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport abs as c_abs
from scipy.optimize import linear_sum_assignment
import time
from graph_pkg.graph.edge cimport Edge

cdef class GED:

    def __cinit__(self, EditCost edit_cost):
        self.edit_cost = edit_cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _create_c_matrix(self):
        cdef:
            int i, j
            double cost

        self.C = np.zeros((self._n + self._m, self._n + self._m),
                          dtype=np.float64)

        # Create substitute part
        for i in range(self._n):
            for j in range(self._m):
                cost = self.edit_cost.cost_substitute_node(self.graph_source.nodes[i],
                                                           self.graph_target.nodes[j])
                print(f'{i} - {j}: cost {cost}')
                self.C[i][j] = cost

        # Create node deletion part
        self.C[0:self._n:1, self._m:self._n+self._m:1] = np.inf
        for i in range(self._n):
            j = self._m + i
            cost = self.edit_cost.cost_delete_node(self.graph_source.nodes[i])

            self.C[i][j] = cost

        # Create node insertion part
        self.C[self._n:self._n+self._m:1, 0:self._m:1] = np.inf
        for j in range(self._m):
            i = self._n + j
            cost = self.edit_cost.cost_insert_node(self.graph_target.nodes[j])

            self.C[i][j] = cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _create_c_star_matrix(self):
        cdef:
            int i, j
            double cost
            int[::1] out_degrees_source, out_degrees_target


        out_degrees_source = self.graph_source.out_degrees()
        out_degrees_target = self.graph_target.out_degrees()

        self.C_star = np.zeros((self._n + self._m, self._n + self._m),
                               dtype=np.float64)
        self.C_star[::1, ::1] = self.C

        # Update the substitution part
        for i in range(self._n):
            for j in range(self._m):
                cost = c_abs(out_degrees_source[i] - out_degrees_target[j])
                self.C_star[i][j] += cost

        # Update the deletion part
        for i in range(self._n):
            j = self._m + i
            self.C_star[i][j] += out_degrees_source[i]

        # Update the insertion part
        for j in range(self._m):
            i = self._n + j
            self.C_star[i][j] += out_degrees_target[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_cost_node_edit(self, int[::1] phi):
        cdef:
            int i
            double cost = 0.

        for i in range(self._n + self._m):
            cost += self.C[i][phi[i]]

        return cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_cost_edge_edit(self, int[::1] phi):
        cdef:
            int i, j
            double cost = 0.
            Edge edge_source, edge_target

        for i in range(self._n + self._m):
            for j in range(i + 1, self._n + self._m):
                phi_i = phi[i]
                phi_j = phi[j]

                print(f'i {i}, j {j}')

                if self.graph_source.has_edge(i, j):
                    # check for edge substitution
                    if self.graph_target.has_edge(phi_i, phi_j):
                        edge_source = self.graph_source.get_edge_by_node_idx(i, j)
                        edge_target = self.graph_target.get_edge_by_node_idx(phi_i, phi_j)

                        cost += self.edit_cost.cost_substitute_edge(edge_source, edge_target)
                        print(f'-Exchange edge {(i, j)} --> {(phi_i, phi_j)}')


                    #check for edge deletion
                    else:
                        edge_source = self.graph_source.get_edge_by_node_idx(i, j)

                        cost += self.edit_cost.cost_delete_edge(edge_source)

                        print(f'#deletion edge {(i, j)} --> empty')
                else:
                    # check for edge insertion
                    if self.graph_target.has_edge(phi_i, phi_j):
                        edge_target = self.graph_target.get_edge_by_node_idx(phi_i, phi_j)

                        cost += self.edit_cost.cost_insert_edge(edge_target)

                        print(f'*insertion edge empty --> {(phi_i, phi_j)}')

        return cost

    cpdef double compute_distance_between_graph(self, Graph graph_source, Graph graph_target):
        cdef:
            int i, j
            int[::1] phi
            double edit_cost = 0.

        self.graph_source = graph_source
        self.graph_target = graph_target
        self._n = len(self.graph_source)
        self._m = len(self.graph_target)

        self._create_c_matrix()
        self._create_c_star_matrix()

        _, col_ind = linear_sum_assignment(self.C_star)
        phi = col_ind.astype(dtype=np.int32)
        print(col_ind.dtype)

        edit_cost += self._compute_cost_node_edit(phi)
        edit_cost += self._compute_cost_edge_edit(phi)

        return edit_cost
