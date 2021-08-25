import numpy as np
cimport numpy as np
from collections import Counter
import os
from pathlib import Path

cdef class KNNLinearCombination:
    """
    Basic KNN classifier.
    Look for the k closest neighbors of a data point.
    Returns the majority class of the k neighbors.
    """

    def __init__(self, GED ged, int k, bint parallel=False):
        """
        Initialize the KNN with the graph edit distance class.
        The ged is used to compute the distance between 2 graphs.

        :param ged: GED
        """
        self.ged = ged
        self.k = k
        self.mat_dist = MatrixDistances(ged, parallel)
        self.are_distances_loaded = False


    cdef void _init_folder_distances(self, str folder_distances, bint is_test_set=False):
        if folder_distances:
            directory = f'saved_distances{"_test" if is_test_set else "_val"}'
            self.folder_distances = os.path.join(folder_distances, directory)
            Path(self.folder_distances).mkdir(parents=True, exist_ok=True)
        else:
            self.folder_distances = ''

    cpdef void train(self, HierarchicalGraphs h_graphs_train, list labels_train):
        """
        Function used to train the KNN classifier.
        With KNN, the training step consists to save the training set.
        The saved training set is used later during the prediction step.

        :param X_train: list
        :param y_train: list
        :return: 
        """
        self.h_graphs_train = h_graphs_train
        self.labels_train = labels_train
        self.np_labels_train = np.array(labels_train, dtype=np.int32)

    cpdef void load_h_distances(self, HierarchicalGraphs h_graphs_pred,
                                str folder_distances='', bint is_test_set=False, int num_cores=-1):
        cdef:
            int size_pred_set
            tuple shape

        self._init_folder_distances(folder_distances, is_test_set=is_test_set)

        size_pred_set = len(h_graphs_pred.hierarchy[1.0])
        self.percent_hierarchy = list(h_graphs_pred.hierarchy.keys())
        shape = (len(self.percent_hierarchy), len(self.labels_train), size_pred_set)
        self.h_distances = np.empty(shape)

        file = os.path.join(self.folder_distances, f'h_distances.npy')

        if os.path.isfile(file):
            print('\nThe distances are already computed\nLoaded from file\n')
            with open(file, 'rb') as f:
                self.h_distances = np.load(f)
        else:
            print('\nCompute the distances\n')
            for idx, percentage in enumerate(self.percent_hierarchy):
                    self.h_distances[idx] = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[percentage],
                                                                           h_graphs_pred.hierarchy[percentage],
                                                                           heuristic=True,
                                                                           num_cores=num_cores)
            if self.folder_distances:
                with open(file, 'wb') as f:
                    np.save(f, self.h_distances)

        self.are_distances_loaded = True


    cpdef int[::1] predict_dist(self, double[::1] omegas):
        """
        With this method, all the distance matrices are merged into one distance matrix.
        The matrices are merged by summing them up together.
        The importance of each distance matrix is given by the vector omegas.
        
        The final distance matrix is obtained like:
            D_final = O.T D
                    = o_1·d_1 + o_2·d_2 + ... + o_n·d_n
                    
        Once D_final is obtained, the classification is done as in a standard KNN model.
        
        :param omegas: Vector that weight the strengt of the distance matrices
        :return: 
        """
        cdef:
            int dim1, dim2
            int[::1] predictions
            double[::1] normalized_omegas
            double[:, ::1] combination_distances

        assert self.are_distances_loaded, f'Please call first load_h_distances().'

        # Create distances matrix
        dim1, dim2 = self.h_distances.shape[1:3]
        combination_distances = np.zeros((dim1, dim2))

        # Normalize the omega vector to have the o_i \in [0,1]
        normalized_omegas = omegas / np.sum(omegas)

        # Summing up the distance matrices
        for idx, _ in enumerate(self.percent_hierarchy):
            combination_distances += np.array(self.h_distances[idx, :, :]) * normalized_omegas[idx]

        # Get the index of the k smallest distances in the matrix distances.
        idx_k_nearest = np.argpartition(combination_distances, self.k, axis=0)[:self.k]

        # Get the label of the k smallest distances.
        labels_k_nearest = np.asarray(self.np_labels_train)[idx_k_nearest]

        # Get the array of the prediction of all the elements of X.
        predictions = np.array([Counter(arr).most_common()[0][0]
                                   for arr in labels_k_nearest.T])

        return predictions

    cpdef int[:, ::1] predict_score(self):
        """
        Predict the class for all the hierarchy levels.
        The predictions of each level are then combined for the final prediction.
        :return: 
        """
        cdef:
            int dim1, dim2
            int[::1] h_predictions, predictions
            list overall_predictions = []

        assert self.are_distances_loaded, f'Please call first load_h_distances().'

        print(f'-- Start Prediction (score) --')

        for idx, _ in enumerate(self.percent_hierarchy):

            # Get the index of the k smallest distances in the matrix distances.
            idx_k_nearest = np.argpartition(self.h_distances[idx, :, :], self.k, axis=0)[:self.k]

            # Get the label of the k smallest distances.
            labels_k_nearest = np.asarray(self.np_labels_train)[idx_k_nearest]

            # Get the array of the prediction of all the elements of X.
            h_predictions = np.array([Counter(arr).most_common()[0][0]
                                       for arr in labels_k_nearest.T])

            overall_predictions.append(h_predictions)

        return np.array(overall_predictions)

    cpdef int[::1] compute_pred_from_score(self, int[:, ::1] overall_predictions, double[::1] omegas):
        """
        Take the predictions from each level and combined them with respect to the weights omega.
        
        :param overall_predictions: 
        :param omegas: 
        :return: 
        """
        # Normalize the omega values to be in [0, 1]
        normalized_omegas = omegas / np.sum(omegas)

        # Create an empty vector for the final predictions of the model
        predictions = -1 * np.ones_like(overall_predictions[0])

        for idx, preds in enumerate(np.array(overall_predictions).T):
            class_accumulation = {}
            for pred, val in zip(preds, normalized_omegas):
                class_accumulation[pred] = class_accumulation.get(pred, 0) + val

            temp = Counter(class_accumulation)
            predictions[idx] = temp.most_common()[0][0]

        return predictions
