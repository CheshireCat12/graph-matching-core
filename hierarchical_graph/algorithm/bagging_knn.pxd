from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs


cdef class BaggingKNN:

    cdef:
        int n_estimators
        int[::1] np_labels_train
        list labels_train, estimators, graphs_estimators, labels_estimators
        list k_per_estimator
        HierarchicalGraphs h_graphs_train

    cpdef void train(self, HierarchicalGraphs h_graphs_train, list labels_train,
                     double percentage_train, int random_ks, bint use_reduced_graphs=*)

    # cpdef tuple predict_GA(self, list graphs_pred, int[::1] ground_truth_labels, int k, int num_cores=*)

    cpdef int[:,::1] predict_overall(self, list graphs_pred, int num_cores=*)

    cpdef tuple predict(self, int[:, ::1] overall_predictions, int[::1] ground_truth_labels)
