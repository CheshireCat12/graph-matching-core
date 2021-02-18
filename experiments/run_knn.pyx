import yaml
from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from graph_pkg.algorithm.knn cimport KNNClassifier
import numpy as np
cimport numpy as np


cpdef void _do_prediction(KNNClassifier knn, list graphs, list labels, int k, str set):
    # Do the prediction
    predictions = knn.predict(graphs, k=k)

    # transform the predictions and the labels to np.array
    predictions = np.asarray(predictions)
    lbls_test = np.array(labels, dtype=np.int32)

    # Count the number of correctly classified element
    correctly_classified = np.sum(predictions == lbls_test)
    accuracy = 100 * (correctly_classified / len(graphs))

    print(f'{set} Accuracy {accuracy}')


cpdef void run_knn(dict parameters):
    cdef:
        CoordinatorClassifier coordinator
        KNNClassifier knn
        int[::1] predictions
        double accuracy

    params_coordinator = parameters['coordinator']
    k = parameters['k']

    coordinator = CoordinatorClassifier(**params_coordinator)
    graphs_train, labels_train = coordinator.train_split(conv_lbl_to_code=True)
    graphs_val, labels_val = coordinator.val_split(conv_lbl_to_code=True)
    graphs_test, labels_test = coordinator.test_split(conv_lbl_to_code=True)

    knn = KNNClassifier(coordinator.ged)
    knn.train(graphs_train, labels_train)

    _do_prediction(knn, graphs_val, labels_val, k, 'Validation')
    _do_prediction(knn, graphs_test, labels_test, k, 'Test')
