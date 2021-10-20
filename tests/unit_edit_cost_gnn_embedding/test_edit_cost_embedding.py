import pytest

from graph_pkg.graph.label.label_node_embedding import LabelNodeEmbedding
from graph_pkg.edit_cost.edit_cost_gnn_embedding import EditCostGNNEmbedding
from graph_pkg.graph.node import Node

from graph_pkg.graph.node import Node
import numpy as np

@pytest.mark.parametrize('e_cost, expected',
                         [
                             ((1., 1., 1., 1., 'euclidean'), 1.),
                             ((1., 1., 1., 1., 'euclidean'), 1.),
                             ((1., 1., 1., 1., 'euclidean'), 1.),
                             ((11., 1., 1., 1., 'euclidean'), 11.),
                             ((1., 1.9, 1.9, 1.9, 'euclidean'), 1.),
                         ])
def test_dirac_enzymes_add_node(e_cost, expected):
    node0 = Node(0, LabelNodeEmbedding(np.ones(5)))

    edit_cost = EditCostGNNEmbedding(*e_cost)

    result = edit_cost.cost_insert_node(node0)

    assert result == expected


@pytest.mark.parametrize('e_cost, expected',
                         [
                             ((1., 1., 1., 1., 'euclidean'), 1.),
                             ((1., 1., 1., 1., 'euclidean'), 1.),
                             ((16., 12., 18., 17., 'euclidean'), 12.),
                         ])
def test_dirac_enzymes_delete_node(e_cost, expected):
    node0 = Node(0, LabelNodeEmbedding(np.ones(5)))

    edit_cost = EditCostGNNEmbedding(*e_cost)

    result = edit_cost.cost_delete_node(node0)

    assert result == expected


@pytest.mark.parametrize('vec1, vec2, e_cost, expected',
                         [
                             ((1., 1., 1., 2.), (1., 1., 1., 2.), (1., 1., 1., 1., 'euclidean'), 0.),
                             ((0, 0, 0), (0, 0, 1), (1., 1., 1., 1., 'euclidean'), 1.),
                             ((4., 4., 4., 4.), (1., 1., 1., 1.), (3., 3., 2.5, 1., 'euclidean'), 6.),

                         ])
def test_dirac_enzymes_substitution(vec1, vec2, e_cost, expected):
    node0 = Node(0, LabelNodeEmbedding(np.array(vec1, dtype=np.float64)))
    node1 = Node(1, LabelNodeEmbedding(np.array(vec2, dtype=np.float64)))

    edit_cost = EditCostGNNEmbedding(*e_cost)

    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected