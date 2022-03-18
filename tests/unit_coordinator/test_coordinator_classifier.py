import os

import pytest

from graph_pkg_core.coordinator.coordinator_vector_classifier import CoordinatorVectorClassifier

FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           '../test_data/proteins_test')
FOLDER_DATA_OLD = os.path.join(os.path.dirname(__file__),
                               '../test_data/proteins_old')


###### Train #########

@pytest.mark.parametrize('load_method, folder_dataset, expected_size, expected_names, expected_cls, filename',
                         [
                             ('train_split',
                              FOLDER_DATA,
                              10,
                              ['gr_0.graphml', 'gr_2.graphml', 'gr_5.graphml', 'gr_6.graphml', 'gr_8.graphml',
                               'gr_664.graphml', 'gr_665.graphml', 'gr_666.graphml', 'gr_667.graphml',
                               'gr_669.graphml'],
                              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                              None),
                             ('train_split',
                              FOLDER_DATA_OLD,
                              4,
                              ['gr_0.graphml', 'gr_1.graphml', 'gr_2.graphml', 'gr_3.graphml'],
                              [1, 0, 0, 1],
                              None),
                             ('val_split',
                              FOLDER_DATA,
                              6,
                              ['gr_3.graphml', 'gr_22.graphml', 'gr_23.graphml', 'gr_671.graphml', 'gr_672.graphml',
                               'gr_676.graphml'],
                              [0, 0, 0, 1, 1, 1],
                              None),
                             ('test_split',
                              FOLDER_DATA,
                              6,
                              ['gr_1.graphml', 'gr_4.graphml', 'gr_7.graphml', 'gr_663.graphml', 'gr_668.graphml',
                               'gr_673.graphml'],
                              [0, 0, 0, 1, 1, 1],
                              None),
                             ('test_split',
                              FOLDER_DATA_OLD,
                              3,
                              ['gr_889.graphml', 'gr_890.graphml', 'gr_891.graphml'],
                              [0, 0, 1],
                              None),
                             ('test_split',
                              FOLDER_DATA_OLD,
                              3,
                              ['gr_891.graphml', 'gr_889.graphml', 'gr_890.graphml'],
                              [1, 0, 0],
                              'test_reverse'),
                         ])
def test_train_split(load_method, folder_dataset, expected_size, expected_names, expected_cls, filename):
    coordinator = CoordinatorVectorClassifier('proteins',
                                              (1., 1., 1., 1., 'euclidean'),
                                              folder_dataset)
    if filename is not None:
        print(filename)
        X_train, y_train = getattr(coordinator, load_method)(filename)
    else:
        X_train, y_train = getattr(coordinator, load_method)()

    dict_ = {name: lbl for name, lbl in zip(expected_names, expected_cls)}

    assert len(X_train) == expected_size
    assert len(y_train) == expected_size
    for idx, (graph, lbl) in enumerate(zip(X_train, y_train)):
        assert dict_[graph.name] == lbl
