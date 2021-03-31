from graph_pkg.algorithm.graph_edit_distance cimport GED
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs
from graph_pkg.algorithm.matrix_distances cimport MatrixDistances
from graph_pkg.graph.graph cimport Graph

cdef class CoarseToFine:

    cdef:
        HierarchicalGraphs h_graphs_train
        int[::1] labels_train
        GED ged
        MatrixDistances mat_dist

    cpdef void train(self, HierarchicalGraphs h_graphs_train,
                    list labels_train)

    cpdef int[::1] predict(self, HierarchicalGraphs h_graphs_pred,
                           int k)

    cpdef int[::1] predict_percent(self, HierarchicalGraphs h_graphs_pred,
                                   int k, double percent_remaining_graphs=*)