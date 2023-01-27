from cyged.graph_pkg_core.algorithm.graph_edit_distance cimport GED
from cyged.graph_pkg_core.graph.graph cimport Graph


cdef class MatrixDistances:

    cdef:
        bint verbose
        bint parallel
        GED ged


    cpdef double[:, ::1] calc_matrix_distances(self,
                                               list graphs_train,
                                               list graphs_test,
                                               bint heuristic=*,
                                               int num_cores=*)

    cpdef double[:, ::1] _serial_calc_matrix_distances(self,
                                                       list graphs_train,
                                                       list graphs_test,
                                                       bint heuristic=*)

    cpdef double[:, ::1] _parallel_calc_matrix_distances(self,
                                                         list graphs_train,
                                                         list graphs_test,
                                                         bint heuristic=*,
                                                         int num_cores=*)

    cpdef double[::1] do_parallel_computation(self,
                                    list prods,
                                    bint heuristic=*,
                                    int num_cores=*)

    cpdef double _helper_parallel(self, Graph graph_train, Graph graph_test, bint heuristic=*)
