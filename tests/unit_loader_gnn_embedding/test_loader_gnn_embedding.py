import pytest
from graph_pkg.loader_gnn_embedding.loader_gnn_embedding_base import LoaderGNNEmbeddingBase



############## Base ##############

@pytest.mark.parametrize('folder, num_graphs',
                         [('./data_gnn/reduced_graphs_ENZYMES/data/', 600)])
def test_base_loader_embedding(folder, num_graphs):
    loader_base = LoaderGNNEmbeddingBase(folder)
    graphs = loader_base.load()

    assert len(graphs) == num_graphs
