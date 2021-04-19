import numpy as np
cimport numpy as np
from collections import Counter
from itertools import product
from progress.bar import Bar
from graph_pkg.utils.constants cimport PERCENT_HIERARCHY
import os
from pathlib import Path
from collections import defaultdict

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
        shape = (len(PERCENT_HIERARCHY), len(self.labels_train), size_pred_set)
        self.h_distances = np.empty(shape)

        file = os.path.join(self.folder_distances, f'h_distances.npy')

        if os.path.isfile(file):
            print('\nThe distances are already computed\nLoaded from file\n')
            with open(file, 'rb') as f:
                self.h_distances = np.load(f)
        else:
            print('\nCompute the distances\n')
            for idx, percentage in enumerate(PERCENT_HIERARCHY):
                    self.h_distances[idx] = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[percentage],
                                                                           h_graphs_pred.hierarchy[percentage],
                                                                           heuristic=True,
                                                                           num_cores=num_cores)
            if self.folder_distances:
                with open(file, 'wb') as f:
                    np.save(f, self.h_distances)

        self.are_distances_loaded = True


    cpdef int[::1] predict_dist(self, double[::1] omegas):
        cdef:
            int dim1, dim2
            int[::1] predictions
            double[::1] normalized_omegas
            double[:, ::1] combination_distances

        assert self.are_distances_loaded, f'Please call first load_h_distances().'

        # print(f'-- Start Prediction --')
        # Create distances matrix
        dim1, dim2 = self.h_distances.shape[1:3]
        combination_distances = np.zeros((dim1, dim2))
        normalized_omegas = omegas / np.sum(omegas)
        for idx, _ in enumerate(PERCENT_HIERARCHY):
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
        cdef:
            int dim1, dim2
            int[::1] h_predictions, predictions
            list overall_predictions = []

        assert self.are_distances_loaded, f'Please call first load_h_distances().'

        print(f'-- Start Prediction (score) --')

        for idx, _ in enumerate(PERCENT_HIERARCHY):

            # Get the index of the k smallest distances in the matrix distances.
            idx_k_nearest = np.argpartition(self.h_distances[idx, :, :], self.k, axis=0)[:self.k]

            # Get the label of the k smallest distances.
            labels_k_nearest = np.asarray(self.np_labels_train)[idx_k_nearest]

            # Get the array of the prediction of all the elements of X.
            h_predictions = np.array([Counter(arr).most_common()[0][0]
                                       for arr in labels_k_nearest.T])

            overall_predictions.append(h_predictions)

        # predictions = np.array([Counter(arr).most_common()[0][0]
        #                         for arr in np.array(overall_predictions).T])

        return np.array(overall_predictions)

    cpdef int[::1] compute_pred_from_score(self, int[:, ::1] overall_predictions, double[::1] omegas):
        normalized_omegas = omegas / np.sum(omegas)
        predictions = -1 * np.ones_like(overall_predictions[0])
        for idx, preds in enumerate(np.array(overall_predictions).T):
            # temp = defaultdict(lambda x: 0)
            temp = {}
            for pred, val in zip(preds, normalized_omegas):
                temp[pred] = temp.get(pred, 0) + val
            # temp = Counter({pred: val for pred, val in zip(preds, normalized_omegas)})

            temp = Counter(temp)
            # print(temp.most_common()[0][0])
            # print(Counter(preds).most_common()[0][0])
            #
            # print(temp)
            # print(Counter(preds))
            #
            # print(preds)
            # print(np.asarray(omegas))
            #
            # print("####")
            predictions[idx] = temp.most_common()[0][0]
            # preds
            # omegas
            # break
        # predictions = np.array([Counter({pred: val for pred, val in zip(preds, normalized_omegas)}).most_common()[0][0]
        #                         for preds in np.array(overall_predictions).T])
        return predictions


    # cpdef tuple optimize(self, HierarchicalGraphs h_graphs_pred,
    #                      list labels_pred,
    #                      int k,
    #                      str optimization_strategy='linear',
    #                      int num_cores=-1):
    #     """
    #     Predict the class for the graphs in X.
    #     It returns the majority of the k nearest neighbor from the trainset.
    #
    #     - First it computes the distance matrix between the training set and the test set.
    #     - Second it finds the k nearest points for the graphs in the test set.
    #     - Take the majority class from the k nearest point of the training set.
    #
    #     :param graphs_pred: list
    #     :param k: int
    #     :return: predictions: list
    #     """
    #     cdef:
    #         list alphas
    #         int[::1] np_labels_pred
    #         double[:, ::1] distances, tmp_distances
    #         double[:, :, ::1] h_distances
    #
    #     print('\n-- Start prediction --')
    #     np_labels_pred = np.array(labels_pred, dtype=np.int32)
    #
    #     h_distances = self._get_distances(h_graphs_pred, len(labels_pred), num_cores=num_cores)
    #
    #
    #     if optimization_strategy == 'linear':
    #         alphas = [i/10 for i in range(1, 10)]
    #         population = np.array(list(product(alphas, repeat=len(PERCENT_HIERARCHY))))
    #         accuracies = self.fitness(population, h_distances, np_labels_pred, k)
    #
    #         idx_best_acc = np.argmax(accuracies)
    #
    #         return accuracies[idx_best_acc], population[idx_best_acc]
    #
    #     elif optimization_strategy == 'genetic':
    #         np.random.seed(6)
    #
    #         p_crossover = 0.85
    #         p_mutation = 0.01
    #
    #         size_population = 100
    #         round_val = 3
    #         population = np.random.rand(size_population, len(PERCENT_HIERARCHY))
    #         population = np.around(population, round_val)
    #
    #
    #         best_acc = float('-inf')
    #         best_alphas = None
    #         gamma = 0.5
    #         num_turn = 200
    #         distribution_index = 3
    #         bar = Bar('Processing', max=num_turn)
    #
    #         for _ in range(num_turn):
    #             accuracies = self.fitness(population, h_distances, np_labels_pred, k)
    #
    #             idx_max_acc = np.argmax(accuracies)
    #             if accuracies[idx_max_acc] > best_acc:
    #                 best_acc = accuracies[idx_max_acc]
    #                 best_alphas = population[idx_max_acc]
    #                 print('')
    #                 print(f'## Best of Best {best_acc:.2f}, alpha: {best_alphas}')
    #                 print('')
    #
    #             prob = accuracies / np.sum(accuracies)
    #
    #             # tmp = list(range(len(population)))
    #             idx_parents = np.random.choice(len(population),
    #                                            len(population),
    #                                            p=prob)
    #             # print(idx_parents)
    #             #
    #             # print(sorted(idx_parents))
    #             new_population = []
    #
    #             for idx in range(0, len(population), 2):
    #                 idx_parent1, idx_parent2 = idx_parents[idx], idx_parents[idx+1]
    #                 parent1, parent2 = population[idx_parent1], population[idx_parent2]
    #                 child1, child2 = np.zeros(parent1.shape), np.zeros(parent2.shape)
    #
    #                 # Crossover
    #                 for idx_gene, (gene1, gene2) in enumerate(zip(parent1,parent2)):
    #                     # u = np.random.rand()
    #                     # if u <= 0.5:
    #                     #     beta = np.power(2*u, 1/(distribution_index + 1))
    #                     # else:
    #                     #     beta = np.power(1/(2*(1-u)), 1/(distribution_index + 1))
    #                     # new_gene1 = 0.5 * ((1 + beta) * gene1 + (1 - beta) * gene2)
    #                     # new_gene2 = 0.5 * ((1 - beta) * gene1 + (1 + beta) * gene2)
    #                     # child1[idx_gene] = min(max(new_gene1, 0.), 1.0)
    #                     # child2[idx_gene] = min(max(new_gene2, 0.), 1.0)
    #
    #
    #                     low_range = max(min(gene1, gene2) - gamma * abs(gene1 - gene2), 0.0)
    #                     high_range = min(max(gene1, gene2) + gamma * abs(gene1 - gene2), 1.0)
    #                     child1[idx_gene] = np.random.uniform(low=low_range, high=high_range)
    #                     child2[idx_gene] = np.random.uniform(low=low_range, high=high_range)
    #
    #
    #                     # child1[idx_gene] = gamma * gene1 + (1-gamma) * gene2
    #                     # child2[idx_gene] = gamma * gene1 + (1-gamma) * gene2 # (1-gamma) * gene1 + gamma * gene2
    #
    #                     # if np.random.randint(2) == 0:
    #                     #     child1[idx_gene] = gene1
    #                     #     child2[idx_gene] = gene2
    #                     # else:
    #                     #     child1[idx_gene] = gene2
    #                     #     child2[idx_gene] = gene1
    #
    #                     #mutation
    #                     if np.random.rand() <= p_mutation:
    #                         # child1[idx_gene] = np.random.rand()
    #                         child1[idx_gene] = np.random.uniform(low=low_range, high=high_range)
    #                     if np.random.rand() <= p_mutation:
    #                         # child2[idx_gene] = np.random.rand()
    #                         child2[idx_gene] = np.random.uniform(low=low_range, high=high_range)
    #
    #                 new_population.append(child1)
    #                 new_population.append(child2)
    #
    #             population = np.around(np.array(new_population), round_val)
    #
    #             bar.next()
    #         bar.finish()
    #
    #         return best_acc, best_alphas
