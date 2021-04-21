from collections import Counter
cimport numpy as np
import numpy as np

from graph_pkg.algorithm.knn cimport KNNClassifier as KNN
from graph_pkg.algorithm.graph_edit_distance cimport GED

cdef class BaggingKNN:

    def __init__(self, int n_estimators, GED ged):
        self.n_estimators = n_estimators
        self.estimators = [KNN(ged, parallel=True) for _ in range(n_estimators)]
        self.graphs_estimators = [[] for _ in range(n_estimators)]
        self.labels_estimators = [[] for _ in range(n_estimators)]

        np.random.seed(42)

    cpdef void train(self, HierarchicalGraphs h_graphs_train, list labels_train):
        cdef:
            int num_samples, idx_estimator, graph_idx, lambda_

        self.h_graphs_train = h_graphs_train
        self.labels_train = labels_train
        self.np_labels_train = np.array(labels_train, dtype=np.int32)

        num_samples = len(labels_train)
        all_graphs = set(range(num_samples))

        lambdas = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        for idx_estimator in range(self.n_estimators):
            graphs_choice = np.random.choice(num_samples, size=num_samples, replace=True)
            out_of_bag = all_graphs.difference(graphs_choice)
            lambda_choice = np.random.choice(lambdas, size=num_samples, replace=True, p=lambdas/np.sum(lambdas))
            # print(lambda_choice)
            for graph_idx, lambda_c in zip(graphs_choice, lambda_choice):
                # print(graph_idx, lambda_c)
                self.graphs_estimators[idx_estimator].append(self.h_graphs_train.hierarchy[lambda_c][graph_idx])
                self.labels_estimators[idx_estimator].append(self.labels_train[graph_idx])

            self.estimators[idx_estimator].train(self.graphs_estimators[idx_estimator],
                                                 self.labels_estimators[idx_estimator])

    cpdef int[::1] predict(self, list graphs_pred, int k):
        overall_predictions = []

        for idx, estimator in enumerate(self.estimators):
            overall_predictions.append(estimator.predict(graphs_pred, k))

        predictions = np.array([Counter(arr).most_common()[0][0]
                                for arr in np.array(overall_predictions).T])

        return predictions