cimport cython

cdef class MatrixDistances:

    def __cinit__(self, list graphs, GED ged):
        self.graphs = graphs
        self.ged = ged

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void create_matrix_distance_diagonal(self):
        cdef:
            int i, j, n
            double edit_cost
            double[:, ::1] distances
            Graph graph_source, graph_target

        distances = np.zeros((i, i), dtype=np.float64)
        for i in range(n):
            graph_source = self.graphs[i]

            for j in range(i + 1, n):
                graph_target = self.graphs[j]

                edit_cost = self.ged.compute_edit_distanc(graph_source,
                                                          graph_target)
                distances[i][j] = edit_cost


