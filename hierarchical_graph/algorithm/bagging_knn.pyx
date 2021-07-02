from collections import Counter, defaultdict
cimport numpy as np
import numpy as np

from progress.bar import Bar

from graph_pkg.algorithm.knn cimport KNNClassifier as KNN
from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.utils.functions.helper import calc_accuracy, calc_f1

cdef class BaggingKNN:

    def __init__(self, int n_estimators, GED ged, int seed=42, int num_cores=-1):
        self.n_estimators = n_estimators
        self.estimators = [KNN(ged, parallel=True) for _ in range(n_estimators)]
        self.graphs_estimators = [[] for _ in range(n_estimators)]
        self.labels_estimators = [[] for _ in range(n_estimators)]
        self.k_per_estimator = []
        self.weights_per_estimator = np.zeros(n_estimators)
        self.num_cores = num_cores

        np.random.seed(seed)

    def proportion(self, labels):
        """Verify if the repartition of the classes is homogenous within the training subsets."""
        nb_ones = np.count_nonzero(labels)
        size = len(labels)
        nb_zeros = size - nb_ones

        proportion_1 = nb_ones / size
        proportion_0 = nb_zeros / size
        print(f'proportion of 0: {nb_zeros}/{size} {proportion_0:.2f}')
        print(f'proportion of 1: {nb_ones}/{size} {proportion_1:.2f}\n')



    cpdef void train(self, HierarchicalGraphs h_graphs_train,
                     list labels_train,
                     double percentage_train,
                     int random_ks,
                     double[::1] lambdas,
                     bint use_reduced_graphs=False):
        cdef:
            int num_samples, idx_estimator, graph_idx
            set all_graphs, out_of_bag
            list weights

        self.h_graphs_train = h_graphs_train
        self.labels_train = labels_train
        self.np_labels_train = np.array(labels_train, dtype=np.int32)

        num_samples = int(len(labels_train) * percentage_train)
        all_graphs = set(range(len(labels_train)))


        if use_reduced_graphs:
            print('Use Reduced graphs')
            lambdas = np.array(lambdas)
        else:
            lambdas = np.array([1.0])

        k_values = [1, 3, 5] #, 7, 9, 11]

        bar = Bar('Training', max=self.n_estimators)

        for idx_estimator in range(self.n_estimators):
            graphs_choice = np.random.choice(len(labels_train), size=num_samples, replace=True)
            lambda_choice = np.random.choice(lambdas, size=num_samples, replace=True) # , p=lambdas[::-1]/np.sum(lambdas))

            out_of_bag = all_graphs.difference(graphs_choice)

            # print(f'lambda choice: {lambda_choice}')

            for idx_graph, lambda_c in zip(graphs_choice, lambda_choice):
                self.graphs_estimators[idx_estimator].append(self.h_graphs_train.hierarchy[lambda_c][idx_graph])
                self.labels_estimators[idx_estimator].append(self.labels_train[idx_graph])

            # print('\nProportion training subset')
            # self.proportion(self.labels_estimators[idx_estimator])

            self.estimators[idx_estimator].train(self.graphs_estimators[idx_estimator],
                                                 self.labels_estimators[idx_estimator])

            # Select the k for the current knn
            if random_ks < 0:
                k = np.random.choice(k_values)
            else:
                k = random_ks

            self.k_per_estimator.append(k)

            self.performance_individual_estimator(idx_estimator, out_of_bag, k)

            bar.next()
        bar.finish()

        # print(self.weights_per_estimator.base)

    cpdef void performance_individual_estimator(self, int current_estimator, set oob_graphs, int k):
        cdef:
            list graphs = []
            list labels = []
            int[::1] np_labels

        # print(len(oob_graphs))

        for idx_graph in list(oob_graphs):
            # print('err1')
            graphs.append(self.h_graphs_train.hierarchy[1.0][idx_graph])
            # print('err2')
            labels.append(self.labels_train[idx_graph])

        # print('Proportion of 1/0 in OOB:')
        # self.proportion(labels)

        # print('test1')
        np_labels = np.array(labels, dtype=np.int32)
        # print('test2')
        predictions = self.estimators[current_estimator].predict(graphs, k, self.num_cores)
        accuracy = calc_accuracy(np_labels, predictions)
        # print('test3')
        # print(f'proportion of 1/0 in predictions:')
        # self.proportion(predictions)

        # print(f'f1 score: {calc_f1(np_labels, predictions):.2f}\n'
        #       f'accuracy: {accuracy:.2f}\n')

        self.weights_per_estimator[current_estimator]= accuracy

    # cpdef tuple predict_GA(self, list graphs_pred, int[::1] ground_truth_labels, int k, int num_cores=-1):
    #     cdef:
    #         int size_population = 50
    #         int num_turn = 50
    #         # int[::1] predictions
    #         int[:, ::1] overall_predictions
    #         double p_mutation = 0.05
    #
    #     population = np.random.choice([0, 1], (size_population, self.n_estimators))
    #     best_acc = float('-inf')
    #     best_omegas = None
    #     best_predictions = None
    #
    #     overall_predictions = self.predict_overall(graphs_pred, k, num_cores)
    #
    #     predictions = np.array([Counter(arr).most_common()[0][0]
    #                             for arr in np.array(overall_predictions).T],
    #                            dtype=np.int32)
    #
    #     acc_before_GA_opt = calc_accuracy(ground_truth_labels,
    #                                     np.array(predictions, dtype=np.int32))
    #     print(f'Acc before the GA optimization: {acc_before_GA_opt:.2f}')
    #
    #     for _ in range(num_turn):
    #
    #         accuracies = np.zeros(size_population)
    #         for idx, gene in enumerate(population):
    #
    #             # quick fix
    #             if sum(gene) == 0:
    #                 gene[0] = 1
    #
    #             predictions = []
    #             for arr in np.array(overall_predictions).T:
    #                 arr_mod = [val for val, activate in zip(arr, gene) if activate]
    #                 predictions.append(Counter(arr_mod).most_common()[0][0])
    #
    #             # predictions = np.array([Counter(arr).most_common()[0][0]
    #             #                         for arr in np.array(overall_predictions).T],
    #             #                        dtype=np.int32)
    #
    #             accuracies[idx] = calc_accuracy(ground_truth_labels,
    #                                             np.array(predictions, dtype=np.int32))
    #
    #         idx_max_acc = np.argmax(accuracies)
    #         if accuracies[idx_max_acc] > best_acc:
    #             best_acc = accuracies[idx_max_acc]
    #             best_omegas = population[idx_max_acc]
    #             best_predictions = predictions
    #             print(f'\n## Best of Best {best_acc:.2f}, omegas: {best_omegas}\n')
    #
    #
    #         accuracies = accuracies - (np.min(accuracies) - 0.01)
    #         prob = accuracies / np.sum(accuracies)
    #
    #         idx_parents = np.random.choice(size_population,
    #                                        size_population,
    #                                        p=prob)
    #         new_population = []
    #
    #         for idx in range(0, size_population, 2):
    #             idx_parent1, idx_parent2 = idx_parents[idx], idx_parents[idx+1]
    #             parent1, parent2 = population[idx_parent1], population[idx_parent2]
    #             pt1, pt2 = sorted(np.random.choice(parent1.shape[0], 2))
    #             child1 = np.concatenate((parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]))
    #             child2 = np.concatenate((parent2[:pt1], parent1[pt1:pt2], parent2[pt2:]))
    #
    #             if np.random.rand() <= p_mutation:
    #                 idx_gene = np.random.choice(child1.shape[0])
    #                 child1[idx_gene] = 1 if child1[idx_gene] == 0 else 0
    #
    #             if np.random.rand() <= p_mutation:
    #                 idx_gene = np.random.choice(child2.shape[0])
    #                 child2[idx_gene] = 1 if child2[idx_gene] == 0 else 0
    #
    #             new_population.append(child1)
    #             new_population.append(child2)
    #
    #
    #         population = np.array(new_population, dtype=np.int32)
    #
    #     return acc_before_GA_opt, best_acc, best_omegas, best_predictions

    cpdef int[:,::1] predict_overall(self, list graphs_pred):
        overall_predictions = []

        bar = Bar('Processing', max=self.n_estimators)
        for idx, estimator in enumerate(self.estimators):
            k_pred = self.k_per_estimator[idx]
            overall_predictions.append(estimator.predict(graphs_pred, k_pred,
                                                         num_cores=self.num_cores))

            bar.next()

        bar.finish()

        return np.array(overall_predictions)


    cpdef tuple predict(self, int[:, ::1] overall_predictions, int[::1] ground_truth_labels):
        cdef:
            int[::1] final_predictions # , predictions_per_estimator
            int[::1] predictions = -1 * np.ones(len(ground_truth_labels), dtype=np.int32)

        for idx_prediction, predictions_per_estimator in enumerate(np.array(overall_predictions).T):
            # print('predictions from all estimators')
            weighted_prediction = defaultdict(int)
            # print(predictions_per_estimator)
            temp = []

            for pred, weight in zip(predictions_per_estimator, self.weights_per_estimator):
                temp.append((pred, weight))
                # print(weight)
                weighted_prediction[pred] += weight
                # print('#########')


            predictions[idx_prediction] = max(weighted_prediction, key=weighted_prediction.get)

            # print(temp)
            # print(weighted_prediction)
            # print(f'ground_truth: {ground_truth_labels[idx_prediction]}')
            # print(f'predicion: {predictions[idx_prediction]:.2f}')
            # print('%%%%%%')


        # predictions = np.array([Counter(arr).most_common()[0][0]
        #                         for arr in np.array(overall_predictions).T])

        accuracy = calc_accuracy(ground_truth_labels, np.array(predictions, dtype=np.int32))
        # print(f'f_1 score: {calc_f1(ground_truth_labels, predictions)}')
        f1_score = calc_f1(ground_truth_labels, predictions)
        return accuracy, f1_score, predictions

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
