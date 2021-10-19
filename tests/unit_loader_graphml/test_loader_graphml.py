import pytest
from graph_pkg.loader_graphml.loader_graphml_base import LoaderGraphMLBase



############## Base ##############

@pytest.mark.parametrize('folder, num_graphs',
                         [('./data_GNN/reduced_graphs_ENZYMES/data/', 600)])
def test_all_letters(folder, num_graphs):
    loader_base = LoaderGraphMLBase(folder)
    graphs = loader_base.load()


    assert len(graphs) == num_graphs
