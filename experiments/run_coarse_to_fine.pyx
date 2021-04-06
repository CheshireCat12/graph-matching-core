import os
from pathlib import Path
import time

import numpy as np

from hierarchical_graph.algorithm.coarse_to_fine cimport CoarseToFine
from graph_pkg.utils.functions.helper import calc_accuracy
from hierarchical_graph.gatherer_hierarchical_graphs import GathererHierarchicalGraphs as GAG
from experiments.runner import Runner


class RunnerCoarseToFine(Runner):

    def __init__(self, parameters):
        super().__init__(parameters)

    def run(self):
        print('Run Coarse To Fine\n')

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


    def _run_pred_val_test(self, validation=True):
        cdef:
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
        num_cores = self.parameters.num_cores

        gag = GAG(coordinator_params, percentages, centrality_measure)

        # Create and train the classifier
        classifier = CoarseToFine(gag.coordinator.ged, parallel)
        classifier.train(gag.h_graphs_train, gag.labels_train)

        start_time = time.time()
        if exp == 'pt3':
            predictions = classifier.predict(gag.h_graphs_test, k, limit, num_cores)
        else:
            predictions = classifier.predict_percent(gag.h_graphs_test, k, limit)
        prediction_time = time.time() - start_time

        np_labels_test = np.array(gag.labels_test, dtype=np.int32)
        accuracy = calc_accuracy(predictions, np_labels_test)

        message = f'\nAccuracy {accuracy} \n'\
                  f'Prediction time: {prediction_time:.3f}\n'

        print(message)

        filename = f'res_{centrality_measure}_{exp}'
        self.save_predictions(predictions, np_labels_test, f'{filename}.npy')
        self.save_stats(message, f'{filename}.txt')


cpdef void run_coarse_to_fine(parameters):
    parameters_tuning = RunnerCoarseToFine(parameters)
    parameters_tuning.run()
