from progress.bar import Bar
import numpy as np
cimport numpy as np
import os
from pathlib import Path
from itertools import product
from time import time
from collections import Counter


from hierarchical_graph.algorithm.bagging_knn cimport BaggingKNN
from graph_pkg.utils.functions.helper import calc_accuracy
from hierarchical_graph.gatherer_hierarchical_graphs cimport GathererHierarchicalGraphs as GAG
from experiments.runner import Runner


class RunnerBaggingKnn(Runner):

    def __init__(self, parameters):
        super(RunnerBaggingKnn, self).__init__(parameters)

    def run(self):
        print('Run KNN with Linear Combination')

        # Set parameters
        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']

        gag = GAG(coordinator_params, percentages, centrality_measure, activate_aggregation=True)

        # num_classifiers = [i * 10 for i in range(2, 20, 2)]
        # percentage_data_train = [1.0, 1.2, 1.4]
        # random_ks = [True, False]
        # random_lambda = [True] # , True]
        #
        # best_acc = float('-inf')
        # best_params = (None, None)
        #
        # best_acc_GA = float('-inf')
        # best_params_GA = (None, None, None)
        #
        #
        # for n_estimators, percentage_train, random_k, random_l in product(num_classifiers,
        #                                                                   percentage_data_train,
        #                                                                   random_ks,
        #                                                                   random_lambda):
        #     message = f'n_estimators: {n_estimators}\n' \
        #               f'percentage_train: {percentage_train}\n' \
        #               f'random_k: {random_k}\n' \
        #               f'random_lambda: {random_l}\n'
        #     print(f'\n###############\n'
        #           f'{message}'
        #           f'############### \n')
        #     classifier = BaggingKNN(n_estimators, gag.coordinator.ged)
        #     classifier.train(gag.h_graphs_train, gag.labels_train, percentage_train, random_l)
        #
        #     if random_k:
        #         k_train = -1
        #     else:
        #         k_train = k
        #
        #     np_labels_val = np.array(gag.labels_val, dtype=np.int32)
        #     acc, acc_GA, omegas, predictions = classifier.predict_GA(gag.graphs_val, np_labels_val,
        #                                                              k=k_train, num_cores=num_cores)
        #
        #     message += f'acc {acc}\n' \
        #                f'acc GA : {acc_GA}\n' \
        #                f'omegas: {omegas}'
        #
        #     self.save_stats(message, 'opt_bagging_reduced_graphs.txt', save_params=False)
        #
        #     # filename = 'predictions' + ('_random' if random_l else '')
        #     # self.save_predictions(np.array(predictions, dtype=np.int32), np_labels_val, f'{filename}.npy')
        #
        #     if acc > best_acc:
        #         best_acc = acc
        #         best_params = (n_estimators, percentage_train, random_k)
        #
        #     if acc_GA > best_acc_GA:
        #         best_acc_GA = acc_GA
        #         best_params_GA = (n_estimators, percentage_train, random_k, omegas)

        print('\nEvaluation on Test set\n')
        # best_n_estimators, best_percentage_train, best_random_k, best_omegas = (120, 1.4, True, None)
        best_omegas = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                       0, 1, 0, 0, 0,
                       1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
                       0, 0, 1, 0, 0,
                       0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                       1, 1, 1, 0, 0,
                       1, 1, 1, 0, 0, 0, 0, 1, 1]

        best_params = (120, 1.4, True)
        best_params_GA = (120, 1.4, True, best_omegas)

        ###############
        ## Prediction on all estimators
        ###############
        best_n_estimators, best_percentage_train, best_random_k = best_params

        if best_random_k:
            k_test = -1
        else:
            k_test = k

        final_classifier = BaggingKNN(best_n_estimators, gag.coordinator.ged)
        final_classifier.train(gag.h_graphs_train, gag.labels_train, best_percentage_train)

        test_predictions = final_classifier.predict_overall(gag.graphs_test, k=k_test)
        final_predictions = np.array([Counter(arr).most_common()[0][0]
                                      for arr in np.array(test_predictions).T],
                                     dtype=np.int32)
        final_acc = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
                                  np.array(final_predictions, dtype=np.int32))


        print(final_acc)


        ###############
        ## Prediction on all estimators
        ###############
        best_n_estimators, best_percentage_train, best_random_k = best_params

        if best_random_k:
            k_test = -1
        else:
            k_test = k

        final_classifier = BaggingKNN(best_n_estimators, gag.coordinator.ged)
        final_classifier.train(gag.h_aggregation_graphs, gag.aggregation_labels, best_percentage_train)

        test_predictions = final_classifier.predict_overall(gag.graphs_test, k=k_test)
        final_predictions = np.array([Counter(arr).most_common()[0][0]
                                      for arr in np.array(test_predictions).T],
                                                          dtype=np.int32)
        final_acc_aggregation = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
                                              np.array(final_predictions, dtype=np.int32))

        print(final_acc_aggregation)

        ##############
        ## Predictions with GA
        ##############

        best_n_estimators_GA, best_percentage_train_GA, best_random_k_GA, best_omegas = best_params_GA

        if best_random_k_GA:
            k_test = -1
        else:
            k_test = k

        final_classifier_GA = BaggingKNN(best_n_estimators_GA, gag.coordinator.ged)
        final_classifier_GA.train(gag.h_graphs_train, gag.labels_train, best_percentage_train_GA)

        test_predictions_GA = final_classifier_GA.predict_overall(gag.graphs_test, k=k_test)
        final_predictions_GA = []

        for arr in np.array(test_predictions_GA).T:
            arr_mod = [val for val, activate in zip(arr, best_omegas) if activate]
            final_predictions_GA.append(Counter(arr_mod).most_common()[0][0])


        final_acc_GA = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
                                     np.array(final_predictions_GA, dtype=np.int32))


        message = f'Acc: {final_acc:.2f}\n' \
                  f'Acc aggregation: {final_acc_aggregation:.2f}\n' \
                  f'Acc GA: {final_acc_GA:.2f}\n' \
                  f'Best parameters\n' \
                  f'\tn_estimators: {best_n_estimators}\n' \
                  f'\tpercentage_train: {best_percentage_train}\n' \
                  f'\trandom_k: {best_random_k}\n' \
                  f'Best Parameters GA\n' \
                  f'\tn_estimators: {best_n_estimators_GA}\n' \
                  f'\tpercentage_train: {best_percentage_train_GA}\n' \
                  f'\trandom_k: {best_random_k_GA}\n' \
                  f'\tomegas: {best_omegas}'

        print(message)

        self.save_stats(message, 'final_bagging_reduced_seed_42.txt')

        # best_n_estimators, best_percentage_train, best_random_k, best_omegas = (120, 1.4, True, None)
        # best_omegas = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        #                0, 1, 0, 0, 0,
        #                1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
        #                0, 0, 1, 0, 0,
        #                0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,
        #                1, 1, 1, 0, 0,
        #                1, 1, 1, 0, 0, 0, 0, 1, 1]
        # test of the implementation
        # acc, omegas, predictions = classifier.predict_GA(gag.graphs_val, np_labels_val,
        #                                                  k=k, num_cores=num_cores)
        #
        # print(acc)

        # final_classifier.train(gag.h_aggregation_graphs, gag.aggregation_labels, best_percentage_train)


    def _run_pred_val_test(self, validation=True):
        pass


cpdef void run_bagging_knn(parameters):
    run_bagging_knn = RunnerBaggingKnn(parameters)
    run_bagging_knn.run()
