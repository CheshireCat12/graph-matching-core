import pytest
from graph_pkg.utils.coordinator.coordinator import Coordinator
from hierarchical_graph.algorithm.page_rank import pagerank_power

def test_pagerank():
    coordinator = Coordinator('letter', (1, 1, 1, 1, 'euclidean'))
    graphs = coordinator.graphs
    graph_0 = graphs[0]
    print(*graphs)

    print(graph_0)
    pagerank_power(graph_0.adjacency_matrix)


def test_pagerank_by_hand():
    pass
