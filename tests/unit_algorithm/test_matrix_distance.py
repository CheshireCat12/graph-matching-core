import numpy as np
import pytest

import os

from graph_pkg_core.coordinator.coordinator import Coordinator
from graph_pkg_core.coordinator.graph_loader import load_graphs
from graph_pkg_core.algorithm.matrix_distances import MatrixDistances

FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           '../test_data/proteins_old')


@pytest.fixture()
def graph_coordinator():
    graphs, lbls = load_graphs(FOLDER_DATA)
    coordinator = Coordinator((1., 1., 1., 1., 'euclidean', 0.8),
                              graphs,
                              lbls)

    return coordinator


@pytest.mark.parametrize('parallel',
                         [False, True])
def test_matrix_distance(graph_coordinator, parallel):
    mat_dist = MatrixDistances(graph_coordinator.ged,
                               parallel=parallel)

    gr_tr = graph_coordinator.graphs[:4]
    gr_te = graph_coordinator.graphs[4:]

    dist = mat_dist.calc_matrix_distances(gr_tr, gr_te, heuristic=True)

    expected_dist = np.array([[22.4, 43.6, 98.6],
                              [44.0391919, 43.44507935, 84.4],
                              [24.45685425, 40.4, 96.2],
                              [26.3254834, 42.2, 97.6]])

    assert np.linalg.norm(dist - expected_dist) < 1e-8
