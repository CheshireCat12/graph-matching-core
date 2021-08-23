from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.algorithm.matrix_distances cimport MatrixDistances


cdef class KNNClassifier:

    cdef:
        bint verbose
        list graphs_train
        list labels_train
        int[::1] np_labels_train
        double[:, ::1] current_distances
        GED ged
        MatrixDistances mat_dist

    cpdef void train(self, list graphs_train, list labels_train)

    cpdef int[::1] predict(self, list graphs_pred, int k, int num_cores=*)
