from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.graph.graph cimport Graph


cdef class MatrixDistances:

    cdef:
        bint parallel
        GED ged


    cpdef double[:, ::1] calc_matrix_distances(self,
                                               list graphs_train,
                                               list graphs_test,
                                               bint heuristic=*)

    cpdef double[:, ::1] _serial_calc_matrix_distances(self,
                                                       list graphs_train,
                                                       list graphs_test,
                                                       bint heuristic=*)

    cpdef double[:, ::1] _parallel_calc_matrix_distances(self,
                                                      list graphs_train,
                                                      list graphs_test,
                                                      bint heuristic=*)

    cpdef double _helper_parallel(self, Graph graph_train, Graph graph_test, bint heuristic=*)

    cpdef double[::1] test_parallel(self,
                                        list prods,
                                        bint heuristic=*)
