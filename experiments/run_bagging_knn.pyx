from progress.bar import Bar
import numpy as np
cimport numpy as np
import os
from pathlib import Path
from itertools import product
from time import time

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

        classifier = BaggingKNN(30, gag.coordinator.ged)
        classifier.train(gag.h_graphs_train, gag.labels_train)
        # gag.graphs_test

        graphs_test, labels_test = gag.coordinator.test_split(conv_lbl_to_code=True)
        predictions = classifier.predict(graphs_test, k=k)

        acc = calc_accuracy(np.array(gag.labels_test, dtype=np.int32), predictions)

        print(acc)




    def _run_pred_val_test(self, validation=True):
        pass


cpdef void run_bagging_knn(parameters):
    run_bagging_knn = RunnerBaggingKnn(parameters)
    run_bagging_knn.run()
