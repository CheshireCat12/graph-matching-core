cimport cython
import numpy as np
cimport numpy as np

from progress.bar import Bar

cdef class MatrixDistances:

    def __cinit__(self, list graphs, GED ged):
        self.graphs = graphs
        self.ged = ged

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:, ::1] create_matrix_distance_diagonal(self):
        cdef:
            int i, j, n
            double edit_cost
            double[:, ::1] distances
            Graph graph_source, graph_target

        n = len(self.graphs)
        bar = Bar('Processing', max=n)
        distances = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            graph_source = self.graphs[i]

            for j in range(n):
                if i == j:
                    continue
                graph_target = self.graphs[j]

                edit_cost = self.ged.compute_edit_distance(graph_source,
                                                          graph_target)
                distances[i][j] = edit_cost
            bar.next()

        bar.finish()

        return distances
