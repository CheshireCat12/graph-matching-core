from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.algorithm.matrix_distances cimport MatrixDistances
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs

cdef class KNNLinearCombination:

    cdef:
        HierarchicalGraphs h_graphs_train
        list labels_train
        int[::1] np_labels_train
        GED ged
        MatrixDistances mat_dist

    cpdef void train(self, HierarchicalGraphs h_graphs_train,
                     list labels_train)

    cpdef double[:, :, ::1] _get_distances(self, HierarchicalGraphs h_graphs_pred,
                                           int size_pred_set)

    cpdef tuple optimize(self, HierarchicalGraphs h_graphs_pred,
                         list labels_pred,
                         int k,
                         str optimization_strategy=*)

    cpdef double predict(self, HierarchicalGraphs h_graphs_pred,
                         list labels_pred,
                         int k,
                         double[::1] alphas,
                         bint save_predictions=*,
                         str folder=*)

    cpdef double[::1] fitness(self, double[:, ::1] population,
                              double[:, :, ::1] h_distances,
                              int[::1] np_labels_test,
                              int k,
                              bint save_predictions=*,
                              str folder=*)