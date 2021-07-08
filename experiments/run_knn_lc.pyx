from hierarchical_graph.algorithm.knn_linear_combination cimport KNNLinearCombination as KNNLC
from experiments.runner import Runner
from hierarchical_graph.gatherer_hierarchical_graphs cimport GathererHierarchicalGraphs as GAG
from graph_pkg.utils.functions.helper import calc_accuracy, calc_f1
from pathlib import Path
import numpy as np
cimport numpy as np
from itertools import product

from progress.bar import Bar
import os


# cpdef void _write_results(double acc, double exec_time, parameters, name):
#     Path(parameters.folder_results).mkdir(parents=True, exist_ok=True)
#     # name = f'percent_remain_{parameters.percentage}_' \
#     #        f'measure_{parameters.centrality_measure}_' \
#     #        f'del_strat_{parameters.deletion_strategy}.txt'
#     filename = os.path.join(parameters.folder_results, name)
#     with open(filename, mode='a+') as fp:
#         fp.write(str(parameters))
#         fp.write(f'\n\nAcc: {acc}; Time: {exec_time}\n'
#                  f'{"="*50}\n\n')
#
# cpdef void _write_results_new(list accuracies, list exec_times, parameters, name):
#     Path(parameters.folder_results).mkdir(parents=True, exist_ok=True)
#
#     filename = os.path.join(parameters.folder_results, name)
#
#     with open(filename, mode='a+') as fp:
#         fp.write(str(parameters))
#         fp.write(f'\n\nAcc: {",".join([str(val) for val in accuracies])};'
#                  f'\nAlphas: {",".join([str(val) for val in exec_times])}\n'
#                  f'{"="*50}\n\n')


class RunnerKnnLC(Runner):

    def __init__(self, parameters):
        super(RunnerKnnLC, self).__init__(parameters)

    def run(self):
        print('Run KNN with Linear Combination')

        if self.parameters.optimize:
            best_params = self.optimization()

            acc_on_test = self.evaluate(best_params)


        else:
            best_params = self.parameters.knn_lc_params.values()

            self.evaluate(best_params)

    def optimization(self):
        """
        Optimize the hyperparmaters of the linear combination.
        Try to find the best combinations of distances to improve the overall accuracy of the model.

        :return:
        """
        cdef:
            KNNLC knn_lc

        ##################
        # Set parameters #
        ##################
        params_edit_cost = self.parameters.coordinator['params_edit_cost']
        best_alpha = self.parameters.best_alpha
        self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)

        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']
        cv = self.parameters.cv

        gag = GAG(coordinator_params, percentages, centrality_measure)

        knn_lc = KNNLC(gag.coordinator.ged, k, parallel)

        #########################################
        # Configure the optimization parameters #
        #########################################

        # Create the range of the coefficient (omega)
        omegas_range = [i / 10 for i in range(0, 10)]
        # omegas_range = np.arange(0, 1, step=0.05)[1:]

        # Create all the combinations of all the coefficients
        # Remove the first element (0,0,0,0,0) (it would lead to an empty distance matrix)
        coefficients = list(product(omegas_range, repeat=len(percentages)))[1:]

        accuracies = np.zeros((len(coefficients), cv))
        f1_scores = np.zeros((len(coefficients), cv))

        for idx, (h_graphs_train, labels_train, h_graphs_val, labels_val) in enumerate(gag.k_fold_validation(cv=cv)):

            # Train the classifier
            knn_lc.train(h_graphs_train, labels_train)

            # Compute the distances in advance not to have to compute it every turn
            knn_lc.load_h_distances(h_graphs_val, folder_distances='',
                                    is_test_set=False,
                                    num_cores=num_cores)

            best_acc = float('-inf')

            if not self.parameters.dist:
                overall_predictions = knn_lc.predict_score()

            bar = Bar(f'Processing, Turn {idx+1}/{cv}', max=len(coefficients))

            for idx_coef, omegas in enumerate(coefficients):
                omegas = np.array(omegas)

                if self.parameters.dist:
                    predictions = knn_lc.predict_dist(omegas)
                else:
                    predictions = knn_lc.compute_pred_from_score(overall_predictions, omegas)

                acc = calc_accuracy(np.array(labels_val, dtype=np.int32), predictions)
                f1_score = calc_f1(np.array(labels_val, dtype=np.int32), predictions)

                accuracies[idx_coef][idx] = acc
                f1_scores[idx_coef][idx] = f1_score

                if acc > best_acc:
                    best_acc = acc

                bar.next()

            print(f'best acc : {best_acc}')
            bar.finish()


        mean_acc = np.mean(accuracies, axis=1)
        median_acc = np.median(accuracies, axis=1)
        mean_f1_score = np.mean(f1_scores, axis=1)

        # print(np.max(mean_acc))
        # print(np.max(median_acc))
        indices_max_mean = mean_acc.argsort()[-10::][::-1]
        indices_max_median = median_acc.argsort()[-10::][::-1]
        # print(mean_acc[indices_max_mean])
        # print(median_acc[indices_max_median])

        idx_best_omega = np.argmax(mean_acc)
        best_omega = np.array(coefficients[idx_best_omega])
        max_mean = mean_acc[idx_best_omega]
        print(max_mean)
        print(max(mean_acc))
        best_coefficients = []

        message = ''

        for idx, (mean, median, f1) in enumerate(zip(mean_acc, median_acc, mean_f1_score)):
            if mean == max_mean:
                message += f'Mean/Median/F1 Score: {mean}, {median}, {f1}\n' \
                           f'{", ".join(f"{val:.2f}" for val in coefficients[idx])}\n' \
                           f'########\n'
                best_coefficients.append(coefficients[idx])

        self.save_stats(message, 'coefficients.txt', save_params=False)


        print(max(mean_acc))
        print(best_omega)

        return best_coefficients

    def evaluate(self, best_params):
        cdef:
            KNNLC knn_lc

        ##################
        # Set parameters #
        ##################
        if not self.parameters.optimize:
            params_edit_cost = self.parameters.coordinator['params_edit_cost']
            best_alpha = self.parameters.best_alpha
            self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)

        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']
        cv = self.parameters.cv

        gag = GAG(coordinator_params, percentages, centrality_measure)

        knn_lc = KNNLC(gag.coordinator.ged, k, parallel)



        # best_omegas, *_ = best_params
        # best_omegas = np.array(best_omegas)
        # print('best_omegas')
        # print(best_omegas)


        # knn.train(gag.h_aggregation_graphs, gag.aggregation_labels)
        knn_lc.train(gag.h_graphs_train, gag.labels_train)
        knn_lc.load_h_distances(gag.h_graphs_test, self.parameters.folder_results,
                             is_test_set=True, num_cores=num_cores)


        if self.parameters.dist:
            pass
            # predictions_final = knn_lc.predict_dist(best_omegas)
            # accuracy_final = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
            #                            predictions_final)
            # message = f'Acc: {accuracy_final:.3f}\n' \
            #           f'Linear combination: {best_omegas}'
        else:
            best_acc = float('-inf')
            message = ''
            for best_omegas in best_params:
                overall_predictions = knn_lc.predict_score()
                predictions_final = knn_lc.compute_pred_from_score(overall_predictions, np.array(best_omegas))

                accuracy_final = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
                                               predictions_final)

                if accuracy_final > best_acc:
                    best_acc = accuracy_final

                message += f'Acc: {accuracy_final:.2f}, Best so far: {best_acc:.2f}\n' \
                           f'Linear combination: {best_omegas}\n'



        filename = f'acc_{"dist" if self.parameters.dist else "score"}'
        print(message)
        self.save_stats(message, name=f'{filename}.txt')

        return accuracy_final


cpdef void run_knn_lc(parameters):
    run_knn_lc = RunnerKnnLC(parameters)
    run_knn_lc.run()
