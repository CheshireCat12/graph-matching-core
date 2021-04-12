from hierarchical_graph.algorithm.knn_linear_combination cimport KNNLinearCombination as KNNLC
from experiments.runner import Runner
from hierarchical_graph.gatherer_hierarchical_graphs cimport GathererHierarchicalGraphs as GAG

from pathlib import Path
import numpy as np
cimport numpy as np
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
            # acc, alphas = self._run_pred_val_test(validation=False)
            # acc, time_pred = self._run_pred_val_test(validation=False)
            #
            # filename = f'{measure}.txt'
            # _write_results_new([acc], list(alphas), self.parameters, filename)


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

        gag = GAG(coordinator_params, percentages, centrality_measure)

        # Create and train the classifier
        knn = KNNLC(gag.coordinator.ged, parallel)
        knn.train(gag.h_graphs_train, gag.labels_train)

        if self.parameters.do_optimization:
            acc, alphas = knn.optimize(gag.h_graphs_val, gag.labels_val, k,
                                       optimization_strategy=self.parameters.optimization_strategy,
                                       num_cores=num_cores)
        else:
            acc = 0
            alphas = np.array(self.parameters.linear_combination)
        # acc = 77.4
        # alphas = np.array([0.647, 0.49, 0.129, 0.361, 0.98])

        print(f'best acc {acc}, best alphas {np.asarray(alphas)}')
        accuray_final = knn.predict(gag.h_graphs_test, gag.labels_test,
                                    k, alphas, save_predictions=True,
                                    folder=self.parameters.folder_results,
                                    num_cores=num_cores)
        print(accuray_final)
        return accuray_final, alphas


cpdef void run_knn_lc(parameters):
    run_knn_lc = RunnerKnnLC(parameters)
    run_knn_lc.run()
