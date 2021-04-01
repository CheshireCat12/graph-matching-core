import os
from pathlib import Path
import time

import numpy as np

from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from hierarchical_graph.algorithm.coarse_to_fine cimport CoarseToFine
from graph_pkg.utils.functions.helper import calc_accuracy
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweenness import Betweenness


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
        measures = self.parameters.hierarchy_params['centrality_measures']

        params_edit_cost = self.parameters.coordinator['params_edit_cost']
        best_alpha = self.parameters.best_alpha
        self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)

        for measure in measures:
            print('+ Tweaking parameters +')
            print(f'+ Measure: {measure} +\n')

            self.parameters.centrality_measure = measure

            self._run_pred_val_test(validation=False)
            # acc, time_pred = self._run_pred_val_test(validation=False)

            # filename = f'{measure}.txt'
            # _write_results_new([acc], list(alphas), self.parameters, filename)
            break


    def _run_pred_val_test(self, validation=True):
        cdef:
            CoordinatorClassifier coordinator
            CoarseToFine classifier

        # Set parameters
        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']
        limit = self.parameters.limit
        exp = self.parameters.exp

        # Retrieve graphs with labels
        coordinator = CoordinatorClassifier(**coordinator_params)
        graphs_train, labels_train = coordinator.train_split(conv_lbl_to_code=True)
        graphs_val, labels_val = coordinator.val_split(conv_lbl_to_code=True)
        graphs_test, labels_test = coordinator.test_split(conv_lbl_to_code=True)

        # Set the graph hierarchical
        measure = self.__MEASURES[centrality_measure]
        h_graphs_train = HierarchicalGraphs(graphs_train, measure, percentage_hierarchy=percentages)
        h_graphs_val = HierarchicalGraphs(graphs_val, measure, percentage_hierarchy=percentages)
        h_graphs_test = HierarchicalGraphs(graphs_test, measure, percentage_hierarchy=percentages)

        # Create and train the classifier
        classifier = CoarseToFine(coordinator.ged, parallel)
        classifier.train(h_graphs_train, labels_train)
        start_time = time.time()
        if exp == 'pt3':
            predictions = classifier.predict(h_graphs_test, k, limit)
        else:
            predictions = classifier.predict_percent(h_graphs_test, k, limit)
        prediction_time = time.time() - start_time

        np_labels_test = np.array(labels_test, dtype=np.int32)
        accuracy = calc_accuracy(predictions, np_labels_test)

        filename = f'res_{centrality_measure}_{exp}'
        save_predictions(predictions, np_labels_test, f'{filename}.npy', self.parameters)
        save_stats(f'Accuracy {accuracy} \n'
                  f'Prediction time: {prediction_time:.3f}\n',
                  f'{filename}.txt',
                   self.parameters)

cpdef void save_predictions(int[::1] predictions, int[::1] labels_test, str name, parameters):
    filename = os.path.join(parameters.folder_results, name)
    with open(filename, 'wb') as f:
        np.save(f, labels_test)
        np.save(f, predictions)


cpdef void save_stats(str message, str name, parameters):
    Path(parameters.folder_results).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(parameters.folder_results, name)

    with open(filename, mode='a+') as fp:
        fp.write(str(parameters))
        fp.write(f'{message}\n'
                 f'{"="*50}\n\n')

cpdef void run_coarse_to_fine(parameters):
    parameters_tuning = HyperparametersTuning(parameters)
    parameters_tuning.run_hierarchy()
