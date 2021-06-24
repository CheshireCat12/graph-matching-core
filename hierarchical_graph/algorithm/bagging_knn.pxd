from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs


cdef class BaggingKNN:

    cdef:
        int n_estimators
        int[::1] np_labels_train
        list labels_train, estimators, graphs_estimators, labels_estimators
        HierarchicalGraphs h_graphs_train

    cpdef void train(self, HierarchicalGraphs h_graphs_train, list labels_train,
                     double percentage_train, bint random_lambda=*)

    cpdef tuple predict_GA(self, list graphs_pred, int[::1] ground_truth_labels, int k, int num_cores=*)

    cpdef int[:,::1] predict_overall(self, list graphs_pred, int k, int num_cores=*)

    cpdef int[::1] predict(self, list graphs_pred, int k, int num_cores=*)
