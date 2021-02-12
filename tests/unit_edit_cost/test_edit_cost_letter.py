import numpy as np
import pytest

from graph_pkg.edit_cost.edit_cost_letter import EditCostLetter
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.node import Node


@pytest.mark.parametrize('coord1, coord2, epsilon',
                         [((0., 0.), (3., 4.), 1e-9),
                          ((-2., 1.5), (8.2, 4.7), 1e-9)])
def test_euclidean_norm(coord1, coord2, epsilon):
    node0 = Node(0, LabelNodeLetter(*coord1))
    node1 = Node(1, LabelNodeLetter(*coord2))

    edit_cost = EditCostLetter(1., 1., 1., 1., 'euclidean')
    result = edit_cost.cost_substitute_node(node0, node1)

    arr1 = np.array(coord1)
    arr2 = np.array(coord2)
    print(arr1 - arr2)
    print(np.linalg.norm(arr1 - arr2))
    assert abs(result - np.linalg.norm(arr1 - arr2)) < epsilon
    assert result == np.linalg.norm(arr1 - arr2)
#     # assert result == np.linalg.norm(arr1 - arr2)
#     # assert abs(result - np.linalg.norm(arr1 - arr2)) == 0.1

@pytest.mark.parametrize('coord1, coord2, epsilon',
                         [((0., 0.), (3., 4.), 1e-9),
                          ((2., 0.), (2., 0.), 1e-9),
                          ((-2., 1.5), (8.2, 4.7), 1e-9)])
def test_manhattan_norm(coord1, coord2, epsilon):
    node0 = Node(0, LabelNodeLetter(*coord1))
    node1 = Node(1, LabelNodeLetter(*coord2))

    edit_cost = EditCostLetter(1., 1., 1., 1., 'manhattan')
    result = edit_cost.cost_substitute_node(node0, node1)
    arr1 = np.array(coord1)
    arr2 = np.array(coord2)

    assert abs(result - np.linalg.norm(arr1 - arr2, 1)) < epsilon
    assert result == np.linalg.norm(arr1 - arr2, 1)
    # assert result == np.linalg.norm(arr1 - arr2)
    # assert abs(result - np.linalg.norm(arr1 - arr2)) == 0.1