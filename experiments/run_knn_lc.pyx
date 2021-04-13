from hierarchical_graph.algorithm.knn_linear_combination cimport KNNLinearCombination as KNNLC
from experiments.runner import Runner
from hierarchical_graph.gatherer_hierarchical_graphs cimport GathererHierarchicalGraphs as GAG
from graph_pkg.utils.functions.helper import calc_accuracy
from pathlib import Path
import numpy as np
cimport numpy as np
from itertools import product

from progress.bar import Bar
from time import time
import os


cpdef void _write_results(double acc, double exec_time, parameters, name):
    Path(parameters.folder_results).mkdir(parents=True, exist_ok=True)
    # name = f'percent_remain_{parameters.percentage}_' \
    #        f'measure_{parameters.centrality_measure}_' \
    #        f'del_strat_{parameters.deletion_strategy}.txt'
    filename = os.path.join(parameters.folder_results, name)
    with open(filename, mode='a+') as fp:
        fp.write(str(parameters))
        fp.write(f'\n\nAcc: {acc}; Time: {exec_time}\n'
                 f'{"="*50}\n\n')

cpdef void _write_results_new(list accuracies, list exec_times, parameters, name):
    Path(parameters.folder_results).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(parameters.folder_results, name)

    with open(filename, mode='a+') as fp:
        fp.write(str(parameters))
        fp.write(f'\n\nAcc: {",".join([str(val) for val in accuracies])};'
                 f'\nAlphas: {",".join([str(val) for val in exec_times])}\n'
                 f'{"="*50}\n\n')


class RunnerKnnLC(Runner):

    def __init__(self, parameters):
        super(RunnerKnnLC, self).__init__(parameters)

    def run(self):
        print('Run KNN with Linear Combination')

        # set parameters to tune
        percentages = self.parameters.hierarchy_params['percentages']
        measures = self.parameters.hierarchy_params['centrality_measures']

        params_edit_cost = self.parameters.coordinator['params_edit_cost']
        best_alpha = self.parameters.best_alpha
        self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)

        for measure in measures:
            print('+ Tweaking parameters +')
            print(f'+ Measure: {measure} +\n')

            self.parameters.centrality_measure = measure

            self._run_pred_val_test(validation=False)

    def _run_pred_val_test(self, validation=True):
        cdef:
            KNNLC knn_lc

        # Set parameters
        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']
        cv = self.parameters.cv

        gag = GAG(coordinator_params, percentages, centrality_measure)

        knn = KNNLC(gag.coordinator.ged, k, parallel)

        # omegas_range = [i / 10 for i in range(1, 10)]
        # coefficients = list(product(omegas_range, repeat=len(percentages)))
        # accuracies = np.zeros((len(coefficients), cv))
        #
        # for idx, (h_graphs_train, labels_train, h_graphs_val, labels_val) in enumerate(gag.k_fold_validation(cv=cv)):
        #     # print(f'Turn {idx+1}/{cv}')
        #
        #     # Train the classifier
        #     knn.train(h_graphs_train, labels_train)
        #     # Compute the distances in advance not to have to compute it every turn
        #     knn.load_h_distances(h_graphs_val, num_cores=num_cores)
        #
        #     bar = Bar(f'Processing, Turn {idx+1}/{cv}', max=len(coefficients))
        #
        #     for idx_coef, omegas in enumerate(coefficients):
        #         predictions = knn.predict_dist(np.array(omegas))
        #         acc = calc_accuracy(np.array(labels_val, dtype=np.int32), predictions)
        #
        #         accuracies[idx_coef][idx] = acc
        #         bar.next()
        #
        #     bar.finish()
        #
        #
        # mean_acc = np.mean(accuracies, axis=1)
        # idx_best_omega = np.argmax(mean_acc)
        # best_omega = np.array(coefficients[idx_best_omega])
        # print(max(mean_acc))
        # print(best_omega)

        best_omega = np.array(self.parameters.linear_combination) # np.array([0.9, 0.3, 0.2, 0.1, 0.1])
        print(best_omega)


        knn.train(gag.h_aggregation_graphs, gag.aggregation_labels)
        # knn.train(gag.h_graphs_train, gag.labels_train)
        knn.load_h_distances(gag.h_graphs_test, self.parameters.folder_results,
                             is_test_set=True)
        predictions_final = knn.predict_score(best_omega)
        accuracy_final = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
                                       predictions_final)
        print(f'{accuracy_final:.2f}')
        # return accuray_final, alphas


cpdef void run_knn_lc(parameters):
    run_knn_lc = RunnerKnnLC(parameters)
    run_knn_lc.run()
