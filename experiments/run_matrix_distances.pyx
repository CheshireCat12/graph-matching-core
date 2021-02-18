from experiments.matrix_distances import MatrixDistances

import numpy as np
import pandas as pd
from time import time

cpdef str _gr_name_to_df_name(str name):
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


cpdef void run_letter():
    cdef:
        double[:, ::1] distances

    coordinator = Coordinator('letter', (0.9, 0.9, 2.3, 2.3, 'euclidean'), './data/Letter/Letter/HIGH/')
    distances = run(coordinator)
    np_distances = np.asarray(distances)

    graph_names = [_gr_name_to_df_name(graph.name) for graph in coordinator.graphs]
    df = pd.DataFrame(data=np_distances, index=graph_names, columns=graph_names)
    df.to_pickle(path='./data/goal/res_letter_cost_node0.9_cost_edge2.3.pkl')



cpdef void run_AIDS():
    cdef:
        double[:, ::1] distances

    coordinator = CoordinatorClassifier('AIDS', (1.1, 1.1, 0.1, 0.1, 'dirac'), './data/AIDS/data/')

    distances = run(coordinator)
    np_distances = np.asarray(distances)

    classes = _get_classes(coordinator)

    graph_names = [_gr_name_AIDS(graph.filename, classes) for graph in coordinator.graphs]
    # print(graph_names)
    # print(len(graph_names))
    df = pd.DataFrame(data=np_distances, index=graph_names, columns=graph_names)
    df.to_pickle(path='./data/goal/res_AIDS_cost_node1.1_cost_edge0.1.pkl')


cpdef void run_mutagenicity():
    cdef:
        double[:, ::1] distances

    coordinator = CoordinatorClassifier('mutagenicity',
                                        (11.0, 11.0, 1.1, 1.1, 'dirac'),
                                        './data/Mutagenicity/data/')

    distances = run(coordinator)
    np_distances = np.asarray(distances)

    classes = _get_classes(coordinator)

    graph_names = [_gr_name_mutagenicity(graph.filename, graph.name, classes)
                   for graph in coordinator.graphs]
    df = pd.DataFrame(data=np_distances, index=graph_names, columns=graph_names)
    df.to_pickle(path='./data/goal/res_mutagenicity_cost_node11.0_cost_edge1.1.pkl')


cpdef double[:, ::1] run(Coordinator coordinator):
    cdef:
        double[:, ::1] distances

    matrix_distances = MatrixDistances(coordinator.graphs, coordinator.ged)

    print('# Start distance computation')
    start_time = time()
    distances = matrix_distances.create_matrix_distance_diagonal()
    print(f'# Computation time: {time() - start_time:3f} s')

    return distances
