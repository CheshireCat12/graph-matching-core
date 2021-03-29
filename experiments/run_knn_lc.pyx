from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from graph_pkg.utils.constants cimport PERCENT_HIERARCHY
from hierarchical_graph.algorithm.knn_linear_combination cimport KNNLinearCombination as KNNLC
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweenness import Betweenness

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


class HyperparametersTuning:

    __MEASURES = {
        'pagerank': PageRank(),
        'betweenness': Betweenness(),
    }

    def __init__(self, parameters):
        self.parameters = parameters

    def run_hierarchy(self):
        print('Run Hierarchy')

        Path(self.parameters.folder_results).mkdir(parents=True, exist_ok=True)

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

            acc, alphas = self._run_pred_val_test(validation=False)
            # acc, time_pred = self._run_pred_val_test(validation=False)

            filename = f'{measure}.txt'
            _write_results_new([acc], list(alphas), self.parameters, filename)


    def _run_pred_val_test(self, validation=True):
        cdef:
            CoordinatorClassifier coordinator
            KNNLC knn_lc

        # Set parameters
        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        parallel = self.parameters.parallel

        # Retrieve graphs with labels
        coordinator = CoordinatorClassifier(**coordinator_params)
        graphs_train, labels_train = coordinator.train_split(conv_lbl_to_code=True)
        graphs_val, labels_val = coordinator.val_split(conv_lbl_to_code=True)
        graphs_test, labels_test = coordinator.test_split(conv_lbl_to_code=True)

        # Set the graph hierarchical
        measure = self.__MEASURES[centrality_measure]
        h_graphs_train = HierarchicalGraphs(graphs_train, measure)
        h_graphs_val = HierarchicalGraphs(graphs_val, measure)
        h_graphs_test = HierarchicalGraphs(graphs_test, measure)


        # k = 3
        # Create and train the classifier
        knn = KNNLC(coordinator.ged, parallel)
        knn.train(h_graphs_train, labels_train)

        if self.parameters.do_optimization:
            acc, alphas = knn.optimize(h_graphs_val, labels_val, k,
                                       optimization_strategy=self.parameters.optimization_strategy)
        else:
            acc = 0
            alphas = np.array(self.parameters.linear_combination)
        # acc = 77.4
        # alphas = np.array([0.647, 0.49, 0.129, 0.361, 0.98])

        print(f'best acc {acc}, best alphas {np.asarray(alphas)}')
        accuray_final = knn.predict(h_graphs_test, labels_test,
                                    k, alphas, save_predictions=True,
                                    folder=self.parameters.folder_results)

        return accuray_final, alphas




cpdef void run_knn_lc(parameters):
    parameters_tuning = HyperparametersTuning(parameters)
    parameters_tuning.run_hierarchy()
