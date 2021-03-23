from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from graph_pkg.utils.constants cimport PERCENT_HIERARCHY
from graph_pkg.algorithm.knn cimport KNNClassifier
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweenness import Betweenness

from pathlib import Path
import numpy as np
cimport numpy as np
from time import time
import os
from itertools import product
from collections import defaultdict
import pandas as pd


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
                 f'\nTime: {",".join([str(val) for val in exec_times])}\n'
                 f'{"="*50}\n\n')

cpdef tuple _do_prediction(KNNClassifier knn, list graphs, list labels, int k, str set_):
    start_time = time()
    # Do the prediction
    predictions = knn.predict(graphs, k=k)
    prediction_time = time() - start_time

    # transform the predictions and the labels to np.array
    predictions = np.asarray(predictions)
    lbls_test = np.array(labels, dtype=np.int32)

    # Count the number of correctly classified element
    correctly_classified = np.sum(predictions == lbls_test)
    accuracy = 100 * (correctly_classified / len(graphs))

    print(f'{set_} Accuracy {accuracy}')
    print(f'Prediction time: {prediction_time:.3f}\n')

    return (accuracy, prediction_time)


class HyperparametersTuning:

    __MEASURES = {
        'pagerank': PageRank(),
        'betweenness': Betweenness(),
    }

    def __init__(self, parameters):
        self.parameters = parameters

    def fine_tune(self):
        print('Finetune the parameters')

        # set parameters to tune
        alpha_start, alpha_end, alpha_step = self.parameters.tuning['alpha']
        alphas = [alpha_step * i for i in range(alpha_start, alpha_end)]
        ks = self.parameters.tuning['ks']

        params_edit_cost = self.parameters.coordinator['params_edit_cost']

        accuracies = defaultdict(list)

        best_acc = float('-inf')
        best_params = (None,)

        for k, alpha in product(ks, alphas):
            print('+ Tuning parameters +')
            print(f'+ alpha: {alpha:.2f}, k: {k} +\n')

            self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, alpha)
            self.parameters.k = k

            acc, _ = self._run_pred_val_test()
            accuracies[k].append(acc)

            if acc > best_acc:
                best_acc = acc
                best_params = (alpha, k)

            # break

        dataframe = pd.DataFrame(accuracies, index=alphas, columns=ks)
        print(dataframe)

        print(f'Best acc on validation: {best_acc}, best params: {best_params}')

        Path(self.parameters.folder_results).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.parameters.folder_results, 'fine_tuning_heuristic.csv')
        dataframe.to_csv(filename)

        best_alpha, best_k = best_params
        self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)
        self.parameters.k = best_k
        acc_test, time_pred = self._run_pred_val_test(validation=False)
        _write_results(acc_test, time_pred, self.parameters, 'accuracy_test_heuristic.txt')

    def run_hierarchy(self):
        print('Run Hierarchy')

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

            acc, time_pred = self._run_pred_val_test(validation=False)

            filename = f'refactor_measure_{measure}_heuristic.txt'

            _write_results_new(acc, time_pred, self.parameters, filename)


    def _run_pred_val_test(self, validation=True):
        cdef:
            CoordinatorClassifier coordinator
            KNNClassifier knn

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


        accuracies, times = [], []
        for percentage in self.parameters.hierarchy_params['percentages']:
            # Create and train the classifier
            knn = KNNClassifier(coordinator.ged, parallel)
            knn.train(h_graphs_train.hierarchy[percentage], labels_train)

            acc, time_pred = _do_prediction(knn, h_graphs_test.hierarchy[percentage], labels_test, k, 'Test')
            accuracies.append(acc)
            times.append(time_pred)

        return accuracies, times
        #
        # # Create the reduced graphs
        # graphs_train_reduced = h_graph.create_hierarchy_percent(graphs_train,
        #                                                         percentage,
        #                                                         deletion_strategy,
        #                                                         verbose=True)
        # if validation:
        #     graphs_val_reduced = h_graph.create_hierarchy_percent(graphs_val,
        #                                                       percentage,
        #                                                       deletion_strategy,
        #                                                       verbose=True)
        # else:
        #     graphs_test_reduced = h_graph.create_hierarchy_percent(graphs_test,
        #                                                        percentage,
        #                                                        deletion_strategy,
        #                                                        verbose=True)
        # Perform prediction
        # if validation:
        #     acc, time_pred = _do_prediction(knn, graphs_val_reduced, labels_val, k, 'Validation')
        # else:
        #     acc, time_pred = _do_prediction(knn, graphs_test_reduced, labels_test, k, 'Test')

        # acc_test, time_test = _do_prediction(knn, graphs_test_reduced, labels_test, k, 'Test')
        #
        # _write_results(acc_val, time_val, self.parameters)
        # _write_results(acc_test, time_test, self.parameters)


cpdef void run_h_knn(parameters):
    parameters_tuning = HyperparametersTuning(parameters)
    if parameters.finetune:
        parameters_tuning.fine_tune()
    else:
        parameters_tuning.run_hierarchy()
