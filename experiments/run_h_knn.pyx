from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from graph_pkg.algorithm.knn cimport KNNClassifier
from hierarchical_graph.hierarchical_graph cimport HierarchicalGraph

from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweeness import Betweeness

from pathlib import Path
import numpy as np
cimport numpy as np
from time import time
import os
from itertools import product


__MEASURES = {
    'pagerank': PageRank(),
    'betweeness': Betweeness(),
}

cpdef void _write_results(double acc, double exec_time, parameters):
    Path(parameters.folder_results).mkdir(parents=True, exist_ok=True)
    name = f'percent_remain_{parameters.percentage}_measure_{parameters.centrality_measure}.txt'
    filename = os.path.join(parameters.folder_results, name)
    with open(filename, mode='a+') as fp:
        fp.write(str(parameters))
        fp.write(f'\n\nAcc: {acc}; Time: {exec_time}\n'
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


cpdef void run_h_knn(parameters):
    cdef:
        CoordinatorClassifier coordinator
        KNNClassifier knn
        int[::1] predictions
        double accuracy

    params_coordinator = parameters.coordinator
    k = parameters.k
    parallel = parameters.parallel

    percentages = [1.0, 0.8, 0.6, 0.4, 0.2]
    measures = ['betweeness', 'pagerank']

    for measure, percentage  in product(measures, percentages):
        print(f'\n{"+"*30}')
        print(f'\n+ Percentage: {percentage}; Measure: {measure} +\n')

        # Init the hyperparameters to test
        parameters.percentage = percentage
        parameters.centrality_measure = measure

        coordinator = CoordinatorClassifier(**params_coordinator)
        graphs_train, labels_train = coordinator.train_split(conv_lbl_to_code=True)
        graphs_val, labels_val = coordinator.val_split(conv_lbl_to_code=True)
        graphs_test, labels_test = coordinator.test_split(conv_lbl_to_code=True)

        measure = __MEASURES[parameters.centrality_measure]

        h_graph = HierarchicalGraph(graphs_train[:3], measure)


        graphs_train_reduced = h_graph.create_hierarchy_percent(graphs_train,
                                                        parameters.percentage,
                                                        verbose=True)
        graphs_val_reduced = h_graph.create_hierarchy_percent(graphs_val,
                                                      parameters.percentage,
                                                      verbose=True)
        graphs_test_reduced = h_graph.create_hierarchy_percent(graphs_test,
                                                       parameters.percentage,
                                                       verbose = True)

        knn = KNNClassifier(coordinator.ged, parallel)
        knn.train(graphs_train_reduced, labels_train)

        acc_val, time_val = _do_prediction(knn, graphs_val_reduced, labels_val, k, 'Validation')
        acc_test, time_test = _do_prediction(knn, graphs_test_reduced, labels_test, k, 'Test')

        _write_results(acc_val, time_val, parameters)
        _write_results(acc_test, time_test, parameters)
