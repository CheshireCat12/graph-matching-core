import os

from graph_pkg_core.coordinator.coordinator_vector import CoordinatorVector

FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           '../test_data/proteins_test')


def test_default_Vector():
    coordinator = CoordinatorVector('proteins',
                                    (1., 1., 1., 1., 'euclidean'),
                                    FOLDER_DATA)

    assert len(coordinator.graphs) == 22
