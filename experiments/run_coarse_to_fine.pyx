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

        coordinator_params = self.parameters.coordinator
        percentages = self.parameters.hierarchy_params['percentages']
        centrality_measure = self.parameters.current_centrality_measure
        gag = GAG(coordinator_params, percentages, centrality_measure)

        for percentage in self.parameters.percentages_to_test:
            self.parameters.percentage_remaining_graphs = percentage
            print('+ Tweaking parameters +')
            print(f'+ Percentage Omega: {percentage} +\n')


            self._run_pred_val_test(gag, validation=False)


    def _run_pred_val_test(self, gag, validation=True):
        cdef:
            CoarseToFine classifier

        # Set parameters
        exp = self.parameters.exp
        k = self.parameters.k
        limit = self.parameters.limit
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        centrality_measure = self.parameters.current_centrality_measure
        percentage_remaining_graphs = self.parameters.percentage_remaining_graphs

        # Create and train the classifier
        classifier = CoarseToFine(gag.coordinator.ged, parallel)
        classifier.train(gag.h_graphs_train, gag.labels_train)

        start_time = time.time()
        if exp == 'pt3':
            predictions, idx_predicted = classifier.predict(gag.h_graphs_test, k, limit, num_cores)
            # predictions, idx_predicted = classifier.predict(gag.h_graphs_val, k, limit, num_cores)
        else:
            predictions = classifier.predict_percent(gag.h_graphs_test, k, limit,
                                                     percentage_remaining_graphs)
            idx_predicted = [0] * len(gag.labels_test)
        prediction_time = time.time() - start_time

        np_labels_test = np.array(gag.labels_test, dtype=np.int32)
        accuracy = calc_accuracy(predictions, np_labels_test)

        # np_labels_val = np.array(gag.labels_val, dtype=np.int32)
        # accuracy = calc_accuracy(predictions, np_labels_val)

        message = f'\nAccuracy {accuracy} \n'\
                  f'Prediction time: {prediction_time:.3f}\n'

        print(message)

        filename = f'res_{centrality_measure}_{exp}'


        self.save_predictions(predictions, np_labels_test, f'{filename}.npy')
        # If needed to compare the accuracy
        # self.save_predictions(np.array(idx_predicted, dtype=np.int32), np_labels_test, f'{filename}_idx_predicted.npy')


        # self.save_predictions(predictions, np_labels_val, f'{filename}.npy')
        # self.save_predictions(np.array(idx_predicted, dtype=np.int32), np_labels_val, f'{filename}_idx_predicted.npy')

        self.save_stats(message, f'{filename}.txt')


cpdef void run_coarse_to_fine(parameters):
    ctf = RunnerCoarseToFine(parameters)
    ctf.run()
