from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.algorithm.matrix_distances cimport MatrixDistances
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs


cdef class KNNLinearCombination:

    cdef:
        GED ged
        int k, num_sub_bunch
        bint are_distances_loaded, augmented_random_graphs
        str folder_distances
        MatrixDistances mat_dist
        HierarchicalGraphs h_graphs_train
        list labels_train, percent_hierarchy
        int[::1] np_labels_train
        double[:, :, ::1] h_distances

    cdef void _init_folder_distances(self, str folder_distances, bint is_test_set=*)

    cpdef void train(self, HierarchicalGraphs h_graphs_train,
                     list labels_train,
                     bint augmented_random_graphs=*,
                     int num_sub_bunch=*)

    cpdef void load_h_distances(self, HierarchicalGraphs h_graphs_pred,
                                str folder_distances=*, bint is_test_set=*, int num_cores=*)

    cpdef void _compute_distances_augmented_random_graphs(self, HierarchicalGraphs h_graphs_pred, int num_cores)

    cpdef void _compute_distances_standard(self, HierarchicalGraphs h_graphs_pred, int num_cores)

    cpdef int[::1] predict_dist(self, double[::1] omegas)

    cpdef int[:, ::1] predict_score(self)

    cpdef int[::1] compute_pred_from_score(self, int[:, ::1] overall_predictions, double[::1] omegas)
