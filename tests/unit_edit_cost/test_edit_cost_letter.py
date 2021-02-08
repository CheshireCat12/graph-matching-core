import numpy as np
import pytest

from graph_pkg.edit_cost.edit_cost_letter_minkowski import EditCostLetterMinkowski
from graph_pkg.graph.node import Node
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter

@pytest.mark.parametrize('coord1, coord2, epsilon',
                         [((0., 0.), (3., 4.), 1e-9),
                          ((-2., 1.5), (8.2, 4.7), 1e-9)])
def test_euclidean_norm(coord1, coord2, epsilon):
    node0 = Node(0, LabelNodeLetter(*coord1))
    node1 = Node(1, LabelNodeLetter(*coord2))

    edit_cost = EditCostLetterMinkowski(2)
    result = edit_cost.cost_substitute_node(node0, node1)

    arr1 = np.array(coord1)
    arr2 = np.array(coord2)
    print(arr1 - arr2)
    print(np.linalg.norm(arr1 - arr2))
    assert abs(result - np.linalg.norm(arr1 - arr2)) < epsilon
    assert result == np.linalg.norm(arr1 - arr2)
    # assert result == np.linalg.norm(arr1 - arr2)
    # assert abs(result - np.linalg.norm(arr1 - arr2)) == 0.1