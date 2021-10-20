import pytest

from graph_pkg.utils.coordinator_gnn_embedding.coordinator_gnn_embedding import CoordinatorGNNEmbedding

def test_default_enzymes():
    coordinator = CoordinatorGNNEmbedding('enzymes', (1., 1., 1., 1., 'euclidean'), './data_gnn/reduced_graphs_ENZYMES/data/')

    assert len(coordinator.graphs) == 600