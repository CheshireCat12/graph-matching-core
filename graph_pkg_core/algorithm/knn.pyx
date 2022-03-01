import numpy as np
cimport numpy as np
from collections import Counter


cdef class KNNClassifier:
    """
    Basic KNN classifier.
    Look for the k closest neighbors of a data point.
    Returns the majority class of the k neighbors.
    """

    def __init__(self, GED ged, bint parallel=False, bint verbose=False):
        """
        Initialize the KNN with the graph edit distance class.
        The ged is used to compute the distance between 2 graphs.

        :param ged: GED
        """
        self.ged = ged
        self.mat_dist = MatrixDistances(ged, parallel, verbose=verbose)
        self.verbose = verbose

    cpdef void train(self, list graphs_train, list labels_train):
        """
        Function used to train the KNN classifier.
        With KNN, the training step consists to save the training set.
        The saved training set is used later during the prediction step.

        :param X_train: list
        :param y_train: list
        :return: 
        """
        self.graphs_train = graphs_train
        self.labels_train = labels_train
        self.np_labels_train = np.array(labels_train, dtype=np.int32)

    cpdef double[:, ::1] compute_dist(self, list graphs_pred, int num_cores=-1):
        return self.mat_dist.calc_matrix_distances(self.graphs_train,
                                                   graphs_pred,
                                                   heuristic=True,
                                                   num_cores=num_cores)

    cpdef int[::1] predict(self, list graphs_pred, int k, int num_cores=-1):
        """
        Predict the class for the graphs in X.
        It returns the majority of the k nearest neighbor from the trainset.
        
        - First it computes the distance matrix between the training set and the test set.
        - Second it finds the k nearest points for the graphs in the test set.
        - Take the majority class from the k nearest point of the training set.
        
        :param graphs_pred: list
        :param k: int
        :return: predictions: list
        """

        if self.verbose:
            print('\n-- Start prediction --')

        # self.current_distances = self.mat_dist.calc_matrix_distances(self.graphs_train,
        #                                                 graphs_pred,
        #                                                 heuristic=True,
        #                                                 num_cores=num_cores)
        self.current_distances = self.compute_dist(graphs_pred, num_cores=num_cores)

        # Get the index of the k smallest distances in the matrix distances.
        idx_k_nearest = np.argpartition(self.current_distances, k, axis=0)[:k]

        # Get the label of the k smallest distances.
        labels_k_nearest = np.asarray(self.np_labels_train)[idx_k_nearest]

        # Get the array of the prediction of all the elements of X.
        prediction = np.array([Counter(arr).most_common()[0][0]
                               for arr in labels_k_nearest.T])

        return prediction