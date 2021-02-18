import yaml
from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from graph_pkg.algorithm.knn cimport KNNClassifier
import numpy as np
cimport numpy as np

cpdef void run_knn():
    cdef:
        CoordinatorClassifier coordinator
        KNNClassifier knn
        int[::1] predictions
        double accuracy

    with open('./configuration/basic_configuration.yml') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    dataset = 'letter'
    params_coordinator = parameters[dataset]['coordinator']
    k = parameters[dataset]['k']

    coordinator = CoordinatorClassifier(**params_coordinator)
    graphs_train, labels_train = coordinator.train_split(conv_lbl_to_code=True)
    graphs_val, labels_val = coordinator.val_split(conv_lbl_to_code=True)
    graphs_test, labels_test = coordinator.test_split(conv_lbl_to_code=True)

    knn = KNNClassifier(coordinator.ged)
    knn.train(graphs_train, labels_train)

    predictions = knn.predict(graphs_val, k=k)
    predictions = np.asarray(predictions)
    lbls_test = np.array(labels_val, dtype=np.int32)
    correctly_classified = np.sum(predictions == lbls_test)
    accuracy = 100 * (correctly_classified / len(graphs_val))
    print(f'Val Accuracy {accuracy}')

    predictions = knn.predict(graphs_test, k=k)
    predictions = np.asarray(predictions)
    lbls_test = np.array(labels_test, dtype=np.int32)
    correctly_classified = np.sum(predictions == lbls_test)
    accuracy = 100 * (correctly_classified / len(graphs_test))
    print(f'Test Accuracy {accuracy}')


