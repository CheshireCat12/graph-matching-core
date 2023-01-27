import numpy as np
import pytest

from cyged import EditCost
from cyged import LabelNode
from cyged import Node


@pytest.mark.parametrize('coord1, coord2, epsilon',
                         [((0., 0.), (3., 4.), 1e-9),
                          ((-2., 1.5), (8.2, 4.7), 1e-9),
                          ((0., 0.), (0.4, 0.3), 1e-9)])
def test_euclidean_norm(coord1, coord2, epsilon):
    node0 = Node(0, LabelNode(np.array(coord1)))
    node1 = Node(1, LabelNode(np.array(coord2)))
    cost_insert = 1.
    edit_cost = EditCost(cost_insert, 1., 1., 1., 'euclidean')
    result = edit_cost.cost_substitute_node(node0, node1)

    arr1 = np.array(coord1)
    arr2 = np.array(coord2)

    expected = min(np.linalg.norm(arr1 - arr2), 2 * cost_insert)
    assert abs(result - expected) < epsilon
