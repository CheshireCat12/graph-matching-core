cdef class KNNClassifier:
    """
    Basic KNN classifier.
    Look for the k closest neighbors of a data point.
    Returns the majority class of the k neighbors.
    """

    def __init__(self, GED ged):
        """
        Initialize the KNN with the graph edit distance class.
        The ged is used to compute the distance between 2 graphs.

        :param ged: GED
        """
        self.ged = ged
        self.mat_dist = MatrixDistances(ged)

    cpdef void train(self, list graphs_train, dict labels_train):
        """
        Function used to train the KNN classifier.
        With KNN, the training step consists to save the training set.
        The saved training set is used later during the prediction step.

        :param X_train: list
        :param y_train: dict
        :return: 
        """
        self.graphs_train = graphs_train
        self.labels_train = labels_train

    cpdef list predict(self, list graphs_pred, int k):
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
        cdef:
            double[:, ::1] distances

        distances = self.mat_dist.calc_matrix_distances(self.graphs_train,
                                                        graphs_pred)

