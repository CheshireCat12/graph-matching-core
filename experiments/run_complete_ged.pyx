import numpy as np
import os
from pathlib import Path
import pandas as pd
from time import time

from graph_pkg.utils.coordinator.coordinator cimport Coordinator
from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from graph_pkg.algorithm.matrix_distances cimport MatrixDistances


cpdef void run_complete_ged(parameters):
    cdef:
        double[:, ::1] distances

    coordinator = CoordinatorClassifier(**parameters.coordinator)
    parallel = parameters.parallel

    distances = run(coordinator, parallel)
    np_distances = np.asarray(distances)

    classes = _get_classes(coordinator)

    dataset = parameters['coordinator']['dataset']
    if dataset == 'letter':
        graph_names = [_gr_name_letter(graph.name)
                       for graph in coordinator.graphs]
    elif dataset == 'AIDS':
        graph_names = [_gr_name_AIDS(graph.filename, classes)
                       for graph in coordinator.graphs]
    elif dataset == 'mutagenicity':
        graph_names = [_gr_name_mutagenicity(graph.filename, graph.name, classes)
                       for graph in coordinator.graphs]
    else:
        raise ValueError('No good dataset!')

    df = pd.DataFrame(data=np_distances, index=graph_names, columns=graph_names)
    Path(parameters['folder_results']).mkdir(parents=True, exist_ok=True)
    df.to_pickle(path=os.path.join(parameters['folder_results'],
                                   f'result_{repr(coordinator.ged.edit_cost)}.pkl'))

    print('\nRun Success!')



cpdef str _gr_name_letter(str name):
    return f'{name[0]}/{name}'


cpdef str _gr_name_AIDS(str name, dict classes):
    return f'{classes[name]}/{name.split(".")[0]}'


cpdef str _gr_name_mutagenicity(str filename, str name, dict classes):
    return f'{classes[filename]}/{name}'


cpdef dict _get_classes(CoordinatorClassifier coordinator):
    # get the classes to create the graph name the same way mathias did it
    X_train, y_train = coordinator.train_split()
    X_test, y_test = coordinator.test_split()
    X_val, y_val = coordinator.val_split()
    Xs = X_train + X_test + X_val
    ys = y_train + y_test + y_val

    classes = {x.filename: lbl for x, lbl in zip(Xs, ys)}
    return classes


cpdef double[:, ::1] run(Coordinator coordinator, bint parallel=False):
    cdef:
        double[:, ::1] distances

    matrix_distances = MatrixDistances(coordinator.ged, parallel)

    print('# Start distance computation')
    start_time = time()
    distances = matrix_distances.calc_matrix_distances(coordinator.graphs,
                                                       coordinator.graphs)
    print(f'# Computation time: {time() - start_time:3f} s')

    return distances
