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

    cpdef int[::1] predict(self, HierarchicalGraphs h_graphs_pred, int k, int limit):
        cdef:
            list graphs_to_pred_with_100 = []
            Graph graph
            double[:, ::1] h_distances_20, h_distances_100

        print('Run Experiment Point 3!\n')

        h_distances_20 = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[0.2],
                                                             h_graphs_pred.hierarchy[0.2],
                                                             heuristic=True)

        # Get the index of the k smallest distances in the matrix distances.
        idx_k_nearest = np.argpartition(h_distances_20, k, axis=0)[:k]

        # Get the label of the k smallest distances.
        labels_k_nearest = np.asarray(self.labels_train)[idx_k_nearest]

        counter_element = 0
        predictions = -1 * np.ones(len(labels_k_nearest.T), dtype=np.int32)
        # Check if the prediction with 20% of the size is "clear"
        for idx, arr in enumerate(labels_k_nearest.T):
            most_common_cls = Counter(arr).most_common()

            if most_common_cls[0][1] >= limit:
                counter_element += 1
                predictions[idx] = most_common_cls[0][0]

        # print(f'Number of clear classification: {counter_element}/{len(labels_k_nearest.T)}, '
        #       f'{100* (counter_element/len(labels_k_nearest.T)):.1f}%')

        idx_still_to_predict, *_ = np.where(predictions < 0)
        # print(len(idx_still_to_predict))
        # print(idx_still_to_predict)

        # find graphs still to predict
        graphs_to_pred_with_100 = [h_graphs_pred.hierarchy[1.0][idx] for idx in idx_still_to_predict]

        # print(len(graphs_to_pred_with_100))

        # print(h_graphs_pred.hierarchy[1.0][-1])
        # print(graphs_to_pred_with_100[-1])

        h_distances_100 = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[1.0],
                                                             graphs_to_pred_with_100,
                                                             heuristic=True)

        # Get the index of the k smallest distances in the matrix distances.
        idx_k_nearest = np.argpartition(h_distances_100, k, axis=0)[:k]


        # Get the label of the k smallest distances.
        labels_k_nearest = np.asarray(self.labels_train)[idx_k_nearest]

        counter_element = 0
        # Check if the prediction with 20% of the size is "clear"
        for idx, arr in zip(idx_still_to_predict, labels_k_nearest.T):
            most_common_cls = Counter(arr).most_common()

            if most_common_cls[0][1] >= limit:
                counter_element += 1

            predictions[idx] = most_common_cls[0][0]

        # print(f'Number of clear classification with 100: {counter_element}/{len(labels_k_nearest.T)}, '
        #       f'{100 * (counter_element / len(labels_k_nearest.T)):.1f}%')
        # print(predictions)


        return predictions


    cpdef int[::1] predict_percent(self, HierarchicalGraphs h_graphs_pred, int k, int limit,
                                   double percent_remaining_graphs=0.1):
        cdef:
            int num_graphs, num_graphs_100
            list graphs_to_pred_with_100 = []
            Graph graph
            double[:, ::1] h_distances_20
            double[:, :] h_distances_100

        print('Run Experiment Point 4!\n')
        # Compute the number of graphs to keep to compute with the original graphs
        num_graphs = len(h_graphs_pred.hierarchy[1.0])
        num_graphs_100 = int(num_graphs * percent_remaining_graphs)
        num_graphs_train = len(self.h_graphs_train.hierarchy[1.0])
        # print(num_graphs_100)

        # Compute distance with 20% of the original size
        h_distances_20 = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[0.2],
                                                             h_graphs_pred.hierarchy[0.2],
                                                             heuristic=True)
        # print(f'shape dist 20{h_distances_20.shape}')
        # Get the index of the k smallest distances in the matrix distances.
        idx_k_nearest = np.argpartition(h_distances_20, k, axis=0)[:k]
        idx_percent_nearest = np.argpartition(h_distances_20, num_graphs_100, axis=0)[:num_graphs_100]

        # Get the label of the k smallest distances.
        labels_k_nearest = np.asarray(self.labels_train)[idx_k_nearest]

        counter_element = 0
        limit = k
        predictions = -1 * np.ones(len(labels_k_nearest.T), dtype=np.int32)
        # Check if the prediction with 20% of the size is "clear"
        for idx, arr in enumerate(labels_k_nearest.T):
            most_common_cls = Counter(arr).most_common()

            if most_common_cls[0][1] >= limit:
                counter_element += 1
                predictions[idx] = most_common_cls[0][0]

        # print(f'Number of clear classification: {counter_element}/{len(labels_k_nearest.T)}, '
        #       f'{100* (counter_element/len(labels_k_nearest.T)):.1f}%')

        idx_still_to_predict, *_ = np.where(predictions < 0)
        # print(len(idx_still_to_predict))

        # find graphs still to predict
        # graphs_pred_100 = [h_graphs_pred.hierarchy[1.0][idx] for idx in idx_still_to_predict]
        # graphs_train_100 = [[self.h_graphs_train.hierarchy[1.0][idx]
        #                      for idx in idx_percent_nearest.T[indices]]
        #                     for indices in idx_still_to_predict]
        # print(idx_percent_nearest.shape)
        m, n = len(idx_still_to_predict), num_graphs_100

        prods = []
        for idx_pred in idx_still_to_predict:
            for idx_train in idx_percent_nearest.T[idx_pred]:
                # print(idx_pred, idx_train)
                prods.append((self.h_graphs_train.hierarchy[1.0][idx_train],
                              h_graphs_pred.hierarchy[1.0][idx_pred]))

        # temp = np.asarray(self.mat_dist.test_parallel(prods, heuristic=True)).reshape((m, n))
        temp = np.asarray(self.mat_dist.test_parallel(prods, heuristic=True)) # .reshape(m, n).T
        # temp = np.zeros((750, m))
        # for idx, idx_pred in enumerate(idx_still_to_predict):
        #     print(h_distances_100.shape)
        #     print(h_distances_100[idx].shape)
        #     temp[idx_pred] = h_distances_100[idx]
        h_distances_100 = np.full((num_graphs_train, m), np.inf)
        idx = 0
        for idx_row, idx_pred in enumerate(idx_still_to_predict):
            for idx_train in idx_percent_nearest.T[idx_pred]:
                # print(idx_pred, idx_train)
                h_distances_100[idx_train][idx_row] = temp[idx]
                idx += 1

        # print(f'@@@@ {h_distances_100.shape}')

        # Get the index of the k smallest distances in the matrix distances.
        idx_k_nearest = np.argpartition(h_distances_100, k, axis=0)[:k]
        # print(idx_k_nearest.T)

        # Get the label of the k smallest distances.
        labels_k_nearest = np.asarray(self.labels_train)[idx_k_nearest]

        # print('----')
        # print(f'shape dist 100 {h_distances_100.shape}')
        # print(f'shape idx nearest {idx_k_nearest.shape}')
        # print(f'shape labels pred {labels_k_nearest.shape}')

        counter_element = 0
        # Check if the prediction with 20% of the size is "clear"
        for idx, arr in zip(idx_still_to_predict, labels_k_nearest.T):
            most_common_cls = Counter(arr).most_common()
            # print(most_common_cls)
            if most_common_cls[0][1] >= limit:
                counter_element += 1

            predictions[idx] = most_common_cls[0][0]

        # print(f'Number of clear classification with 100: {counter_element}/{len(labels_k_nearest.T)}, '
        #       f'{100 * (counter_element / len(labels_k_nearest.T)):.1f}%')
        # print(predictions)


        return predictions