import os
import pickle

from graph_pkg_core.coordinator.coordinator import Coordinator
from graph_pkg_core.coordinator.graph_loader import load_graphs

FOLDER_DATA = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]),
                           'test_data/proteins_test')

#

def test_loader_graphml():
    graphs, _ = load_graphs(FOLDER_DATA, file_extension='graphml')

    assert len(graphs) == 9
    assert len(graphs[0].nodes) == 42
    assert len(graphs[-1].nodes) == 19

def test_loader_pkl():
    graphs_pkl, _ = load_graphs(FOLDER_DATA, file_extension='pkl')

    assert len(graphs_pkl) == 9
    assert len(graphs_pkl[0].nodes) == 42
    assert len(graphs_pkl[-1].nodes) == 19

    # filename = os.path.join(FOLDER_DATA, 'graphs.pkl')
    # with open(filename, 'wb') as f:
    #     pickle.dump(graphs, f)

def test_default_coordinator():
    graphs, _ = load_graphs(FOLDER_DATA, file_extension='graphml')
    coordinator = Coordinator((1., 1., 1., 1., 'euclidean'),
                              graphs,
                              None)
#
    assert len(coordinator.graphs) == 9
    assert len(coordinator.graphs[0].nodes) == 42
    assert len(coordinator.graphs[-1].nodes) == 19
