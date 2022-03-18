import os

import pytest

from graph_pkg_core.loader.loader_train_test_val_split import LoaderTrainTestValSplit

FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           '../test_data/proteins_test')


@pytest.mark.parametrize('load_method, expected_size, expected_names, expected_labels',
                         [
                             ('load_train_split',
                              10,
                              ['gr_0.graphml', 'gr_2.graphml', 'gr_5.graphml', 'gr_6.graphml', 'gr_8.graphml',
                               'gr_664.graphml', 'gr_665.graphml', 'gr_666.graphml', 'gr_667.graphml',
                               'gr_669.graphml'],
                              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                             ('load_val_split',
                              6,
                              ['gr_3.graphml', 'gr_22.graphml', 'gr_23.graphml', 'gr_671.graphml', 'gr_672.graphml',
                               'gr_676.graphml'],
                              [0, 0, 0, 1, 1, 1]),
                             ('load_test_split',
                              6,
                              ['gr_1.graphml', 'gr_4.graphml', 'gr_7.graphml', 'gr_663.graphml', 'gr_668.graphml',
                               'gr_673.graphml'],
                              [0, 0, 0, 1, 1, 1]),

                         ])
def test_split_train(load_method, expected_size, expected_names, expected_labels):
    loader = LoaderTrainTestValSplit(FOLDER_DATA)
    data = getattr(loader, load_method)()

    graph_names = [gr_name for gr_name, _ in data]
    labels = [gr_lbl for _, gr_lbl in data]

    assert len(data) == expected_size
    assert graph_names == expected_names
    assert labels == expected_labels
