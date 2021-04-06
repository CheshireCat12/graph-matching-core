from collections import Counter

import numpy as np
cimport numpy as np


cdef class CoarseToFine:

    def __init__(self, GED ged, bint parallel_computation=True):
        self.ged = ged
        self.mat_dist = MatrixDistances(ged, parallel_computation)

    cpdef void train(self, HierarchicalGraphs h_graphs_train,
                     list labels_train):
        self.h_graphs_train = h_graphs_train
        self.labels_train = np.array(labels_train, dtype=np.int32)

    cpdef void _make_predictions(self, int[::1] predictions, double[:, ::1] h_distances,
                                 int[::1] indices, int k, int limit=-1):
        # Get the index of the k smallest distances in the matrix distances.
        idx_k_nearest = np.argpartition(h_distances, k, axis=0)[:k]

        # Get the label of the k smallest distances.
        labels_k_nearest = np.asarray(self.labels_train)[idx_k_nearest]

        num_clear_decision = 0
        # Check if the predictions predictions are "clear"
        for idx, arr in zip(indices, labels_k_nearest.T):
            most_common_cls = Counter(arr).most_common()

            if most_common_cls[0][1] >= limit:
                predictions[idx] = most_common_cls[0][0]
                num_clear_decision += 1

        if limit > 0:
            print(f'Number of clear classification: {num_clear_decision}/{len(labels_k_nearest.T)}, '
                  f'{100* (num_clear_decision/len(labels_k_nearest.T)):.1f}%')

    cpdef int[::1] predict(self, HierarchicalGraphs h_graphs_pred, int k, int limit, int num_cores=-1):
        cdef:
            int num_predictions
            int[::1] indices, predictions
            double[:, ::1] h_distances_20, h_distances_100
            list graphs_to_pred_with_100 = []
            Graph graph

        print('Run Experiment Point 3!\n')

        num_predictions = len(h_graphs_pred.hierarchy[1.0])

        h_distances_20 = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[0.2],
                                                             h_graphs_pred.hierarchy[0.2],
                                                             heuristic=True,
                                                             num_cores=num_cores)
        indices = np.array(range(num_predictions), dtype=np.int32)

        # Create the prediction vector
        predictions = -1 * np.ones(num_predictions, dtype=np.int32)

        self._make_predictions(predictions, h_distances_20, indices, k, limit=limit)

        # find graphs that need to be predicted with full graphs
        idx_still_to_predict, *_ = np.where(predictions < np.int32(0))

        graphs_to_pred_with_100 = [h_graphs_pred.hierarchy[1.0][idx] for idx in idx_still_to_predict]

        h_distances_100 = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[1.0],
                                                              graphs_to_pred_with_100,
                                                              heuristic=True,
                                                              num_cores=num_cores)

        self._make_predictions(predictions, h_distances_100,
                               idx_still_to_predict.astype(np.int32), k, limit=-1)

        return predictions


    cpdef int[::1] predict_percent(self, HierarchicalGraphs h_graphs_pred, int k, int limit,
                                   double percent_remaining_graphs=0.1):
        cdef:
            int num_graphs, num_graphs_100
            list graphs_to_pred_with_100 = []
            Graph graph
            double[:, ::1] h_distances_20
            double[:, ::1] h_distances_100

        print('Run Experiment Point 4!\n')
        # Compute the number of graphs to keep to compute with the original graphs
        num_predictions = len(h_graphs_pred.hierarchy[1.0])
        num_graphs_100 = int(num_predictions * percent_remaining_graphs)
        num_graphs_train = len(self.h_graphs_train.hierarchy[1.0])

        # Compute distance with 20% of the original size
        h_distances_20 = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[0.2],
                                                             h_graphs_pred.hierarchy[0.2],
                                                             heuristic=True)

        # Retrieve the closest graphs from each predicted graphs
        idx_percent_nearest = np.argpartition(h_distances_20, num_graphs_100, axis=0)[:num_graphs_100]

        predictions = -1 * np.ones(num_predictions, dtype=np.int32)
        indices = np.array(range(num_predictions), dtype=np.int32)

        self._make_predictions(predictions, h_distances_20, indices, k, limit=limit)

        # find graphs still to predict
        idx_still_to_predict, *_ = np.where(predictions < 0)

        # Create the tuples for between the closest graphs and the graphs that remains to be predicted
        prods = []
        for idx_pred in idx_still_to_predict:
            for idx_train in idx_percent_nearest.T[idx_pred]:
                prods.append((self.h_graphs_train.hierarchy[1.0][idx_train],
                              h_graphs_pred.hierarchy[1.0][idx_pred]))

        # Compute the distances between the closest graphs and the remaining graphs
        temp_distances = self.mat_dist.do_parallel_computation(prods, heuristic=True)

        # Create the distance matrix with full graphs
        h_distances_100 = np.full((num_graphs_train, len(idx_still_to_predict)), np.inf)
        idx = 0
        for idx_row, idx_pred in enumerate(idx_still_to_predict):
            for idx_train in idx_percent_nearest.T[idx_pred]:
                h_distances_100[idx_train][idx_row] = temp_distances[idx]
                idx += 1

        self._make_predictions(predictions, h_distances_100,
                               idx_still_to_predict.astype(np.int32), k, limit=-1)

        return predictions