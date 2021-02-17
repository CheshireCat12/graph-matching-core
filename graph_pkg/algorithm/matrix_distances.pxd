from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.graph.graph cimport Graph


cdef class MatrixDistances:

    cdef:
        GED ged


    cpdef double[:, ::1] calc_matrix_distances(self,
                                               list graphs_train,
                                               list graphs_test,
                                               bint heuristic=*)
