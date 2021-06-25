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

        if self.parameters.optimize:
            best_params = self.optimization()
        else:
            best_params = self.parameters.bagging_params.values()

        self.evaluate(best_params)

    def optimization(self):
        """
        Optimize the hyperparameters of the BaggingKNN

        :return: The optimized parameters (n_estimators, percentage_train, random_k, GA_mask)
        """

        ####################
        ## Set parameters ##
        ####################

        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']

        gag = GAG(coordinator_params, percentages, centrality_measure, activate_aggregation=False)

        start, max_estimators, step = self.parameters.fine_tuning['n_estimators']
        num_classifiers = [i for i in range(start, max_estimators+1, step)]
        percentage_data_train = self.parameters.fine_tuning['percentage_data_train']
        random_ks = self.parameters.fine_tuning['random_k']

        # Choose to use the reduced graphs or not
        use_reduced_graphs = True

        best_acc = float('-inf')
        best_params = (None, None, None)

        ########################
        ## Start optimization ##
        ########################

        for percentage_train, random_k in product(percentage_data_train, random_ks):

            if random_k:
                k_train = -1
            else:
                k_train = k

            # 1. Create the BaggingKNN with the maximum number of estimators
            classifier = BaggingKNN(max_estimators, gag.coordinator.ged)
            classifier.train(gag.h_graphs_train, gag.labels_train,
                             percentage_train, k_train, use_reduced_graphs)

            message = f'percentage_train: {percentage_train}\n' \
                      f'random_k: {random_k}\n' \
                      f'reduced_graphs: {use_reduced_graphs}\n'
            print(f'\n###############\n'
                  f'{message}'
                  f'############### \n')


            # 2. Retrieve the predictions of the validation set on all the estimators
            np_labels_val = np.array(gag.labels_val, dtype=np.int32)
            overall_predictions = classifier.predict_overall(gag.graphs_val, num_cores=num_cores)

            # 3. find the best n_estimators
            for n_estimators in num_classifiers:
                preds_n_estimators = overall_predictions[:n_estimators]

                acc_val, predictions = classifier.predict(preds_n_estimators, np_labels_val)

                message += f'n_estimators: {n_estimators}\n' \
                           f'acc {acc_val:.2f}\n\n'

                if acc_val >= best_acc:
                    best_acc = acc_val
                    best_params = (n_estimators, percentage_train, random_k)

            self.save_stats(message, 'opt_bagging_baseline.txt', save_params=False)

        best_n_estimators, best_percentage_train, best_random_k = best_params

        final_msg = f'Best Parameters in validation step\n' \
                    f'Validation Accuracy: {best_acc}\n' \
                    f'\tn_estimators: {best_n_estimators}\n' \
                    f'\tpercentage_train: {best_percentage_train}\n' \
                    f'\trandom_k: {best_random_k}\n'
        print(final_msg)

        return best_params

    # f'n_estimators: {n_estimators}\n' \
    # filename = 'predictions' + ('_random' if random_l else '')
    # self.save_predictions(np.array(predictions, dtype=np.int32), np_labels_val, f'{filename}.npy')

    def evaluate(self, best_params):

        ####################
        ## Set parameters ##
        ####################

        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']

        gag = GAG(coordinator_params, percentages, centrality_measure)

        print('\nEvaluation on Test set\n')

        ###############
        ## Prediction on all estimators
        ###############
        best_n_estimators, best_percentage_train, best_random_k = best_params

        if best_random_k:
            k_test = -1
        else:
            k_test = k

        final_classifier = BaggingKNN(best_n_estimators, gag.coordinator.ged)
        final_classifier.train(gag.h_graphs_train, gag.labels_train,
                               best_percentage_train, k_test, use_reduced_graphs=True)

        test_predictions = final_classifier.predict_overall(gag.graphs_test, num_cores=num_cores)

        final_acc, final_predictions = final_classifier.predict(test_predictions,
                                                                np.array(gag.labels_test, dtype=np.int32))
        # final_predictions = np.array([Counter(arr).most_common()[0][0]
        #                               for arr in np.array(test_predictions).T],
        #                              dtype=np.int32)
        # final_acc = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
        #                           np.array(final_predictions, dtype=np.int32))


        print(f'{final_acc:.2f}')


        ###############
        ## Prediction on all estimators
        ###############
        # best_n_estimators, best_percentage_train, best_random_k = best_params
        #
        # if best_random_k:
        #     k_test = -1
        # else:
        #     k_test = k
        #
        # final_classifier = BaggingKNN(best_n_estimators, gag.coordinator.ged)
        # final_classifier.train(gag.h_aggregation_graphs, gag.aggregation_labels, best_percentage_train)
        #
        # test_predictions = final_classifier.predict_overall(gag.graphs_test, k=k_test)
        # final_predictions = np.array([Counter(arr).most_common()[0][0]
        #                               for arr in np.array(test_predictions).T],
        #                                                   dtype=np.int32)
        # final_acc_aggregation = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
        #                                       np.array(final_predictions, dtype=np.int32))
        #
        # print(final_acc_aggregation)
        final_acc_aggregation = -1

        message = f'Acc test: {final_acc:.2f}\n' \
                  f'Acc aggregation: {final_acc_aggregation:.2f}\n' \
                  f'Best parameters\n' \
                  f'\tn_estimators: {best_n_estimators}\n' \
                  f'\tpercentage_train: {best_percentage_train}\n' \
                  f'\trandom_k: {best_random_k}\n' \

        print(message)

        self.save_stats(message, 'final_bagging_baseline_seed_42.txt')

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
