import os

import pytest

from graph_pkg_core.coordinator.graph_loader import load_graphs

############## VECTOR ##############

FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           '../test_data/proteins_test')


@pytest.mark.parametrize('folder, size_graphs',
                         [
                             (FOLDER_DATA,
                              [42, 27, 10, 24, 11, 336, 108, 154, 19]),
                         ])
def test_all_letters(folder, size_graphs):
    graphs, _ = load_graphs(folder)

    assert len(graphs) == len(size_graphs)
    assert [len(graph) for graph in graphs] == size_graphs


@pytest.mark.parametrize('folder, size_graphs',
                         [
                             (FOLDER_DATA,
                              [42, 27, 10, 24, 11, 336, 108, 154, 19]),
                         ])
def test_loader_pkl(folder, size_graphs):
    graphs, _ = load_graphs(FOLDER_DATA, file_extension='pkl')

    assert len(graphs) == len(size_graphs)
    assert [len(graph) for graph in graphs] == size_graphs
    # filename = os.path.join(FOLDER_DATA, 'graphs.pkl')
    # with open(filename, 'wb') as f:
    #     pickle.dump(graphs, f)
