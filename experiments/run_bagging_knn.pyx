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

        gag = GAG(coordinator_params, percentages, centrality_measure)

        num_classifiers = [i * 10 for i in range(2, 20, 2)]
        percentage_data_train = [1.0, 1.2, 1.4]
        random_ks = [True, False]

        best_acc = float('-inf')
        best_params = (None, None, None)
        # num_classifiers = [3]

        for n_estimators, percentage_train, random_k in product(num_classifiers, percentage_data_train, random_ks):
            message = f'n_estimators: {n_estimators}\n' \
                      f'percentage_train: {percentage_train}\n' \
                      f'random_k: {random_k}\n'
            print(f'\n###############\n'
                  f'{message}'
                  f'############### \n')
            classifier = BaggingKNN(n_estimators, gag.coordinator.ged)
            classifier.train(gag.h_graphs_train, gag.labels_train, percentage_train)

            if random_k:
                k_train = -1
            else:
                k_train = k

            # graphs_val, _ = gag.coordinator.val_split(conv_lbl_to_code=True)
            # print(len(gag.graphs_val))
            np_labels_val = np.array(gag.labels_val, dtype=np.int32)
            acc, omegas, predictions = classifier.predict_GA(gag.graphs_val, np_labels_val,
                                                             k=k_train, num_cores=num_cores)
            # predictions = classifier.predict(graphs_val, k=k_train, num_cores=num_cores)

            # acc = calc_accuracy(np.array(np_labels_val, predictions)

            message += f'acc : {acc}\n' \
                       f'omegas: {omegas}'

            self.save_stats(message, 'temp.txt', save_params=False)

            if acc > best_acc:
                best_acc = acc
                best_params = (n_estimators, percentage_train, random_k, omegas)

        best_n_estimators, best_percentage_train, best_random_k, best_omegas = best_params

        # best_n_estimators, best_percentage_train, best_random_k, best_omegas = (120, 1.4, True, [0, 0])
        final_classifier = BaggingKNN(best_n_estimators, gag.coordinator.ged)
        # final_classifier.train(gag.h_graphs_train, gag.labels_train, best_percentage_train)
        final_classifier.train(gag.h_aggregation_graphs, gag.aggregation_labels, best_percentage_train)

        if best_random_k:
            k = -1

        # best_omegas = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        #                1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        #                0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
        #                1, 1, 1, 0, 0, 0, 0, 1, 1]

        # graphs_test, _ = gag.coordinator.test_split(conv_lbl_to_code=True)
        test_predictions = final_classifier.predict_overall(gag.graphs_test, k=k)
        final_predictions = []
        for arr in np.array(test_predictions).T:
            arr_mod = [val for val, activate in zip(arr, best_omegas)]
            final_predictions.append(Counter(arr_mod).most_common()[0][0])

        acc = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
                            np.array(final_predictions, dtype=np.int32))

        # np_labels_test = np.array(gag.labels_test, dtype=np.int32)
        # acc, omegas, predictions = final_classifier.predict_GA(gag.graphs_test, np_labels_test,
        #                                                        k=k, num_cores=num_cores)

        message = f'Acc: {acc:.2f}\n' \
                  f'best_params\n' \
                  f'\tn_estimators: {best_n_estimators}\n' \
                  f'\tpercentage_train: {best_percentage_train}\n' \
                  f'\trandom_k: {best_random_k}\n' \
                  f'\tomegas: {best_omegas}'

        print(message)

        self.save_stats(message, 'optimization.txt')



    def _run_pred_val_test(self, validation=True):
        pass


cpdef void run_bagging_knn(parameters):
    run_bagging_knn = RunnerBaggingKnn(parameters)
    run_bagging_knn.run()
