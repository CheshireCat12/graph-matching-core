from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.algorithm.matrix_distances cimport MatrixDistances
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs


cdef class KNNLinearCombination:

    cdef:
        GED ged
        int k
        str folder_distances
        MatrixDistances mat_dist
        HierarchicalGraphs h_graphs_train
        list labels_train
        int[::1] np_labels_train
        double[:, :, ::1] h_distances
        bint are_distances_loaded

    cdef void _init_folder_distances(self, str folder_distances, bint is_test_set=*)

    cpdef void train(self, HierarchicalGraphs h_graphs_train,
                     list labels_train)

    cpdef void load_h_distances(self, HierarchicalGraphs h_graphs_pred,
                                str folder_distances=*, bint is_test_set=*, int num_cores=*)

    cpdef int[::1] predict_dist(self, double[::1] omegas)

    cpdef int[:, ::1] predict_score(self)

    cpdef int[::1] compute_pred_from_score(self, int[:, ::1] overall_predictions, double[::1] omegas)

    # cpdef tuple optimize(self, HierarchicalGraphs h_graphs_pred,
    #                      list labels_pred,
    #                      int k,
    #                      str optimization_strategy=*,
    #                      int num_cores=*)
    #
    # cpdef double predict(self, HierarchicalGraphs h_graphs_pred,
    #                      list labels_pred,
    #                      int k,
    #                      double[::1] alphas,
    #                      bint save_predictions=*,
    #                      str folder=*,
    #                      int num_cores=*)
    #
    # cpdef double[::1] fitness(self, double[:, ::1] population,
    #                           double[:, :, ::1] h_distances,
    #                           int[::1] np_labels_test,
    #                           int k,
    #                           bint save_predictions=*,
    #                           str folder=*)