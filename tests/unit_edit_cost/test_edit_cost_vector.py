import numpy as np
import pytest

from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.label.label_hash import LabelHash
from graph_pkg_core.graph.node import Node


@pytest.mark.parametrize('coord1, coord2, epsilon',
                         [((0., 0.), (3., 4.), 1e-9),
                          ((-2., 1.5), (8.2, 4.7), 1e-9),
                          ((0., 0.), (0.4, 0.3), 1e-9)])
def test_euclidean_norm(coord1, coord2, epsilon):
    node0 = Node(0, LabelNodeVector(np.array(coord1)))
    node1 = Node(1, LabelNodeVector(np.array(coord2)))
    cost_insert = 1.
    edit_cost = EditCostVector(cost_insert, 1., 1., 1., 'euclidean')
    result = edit_cost.cost_substitute_node(node0, node1)

    arr1 = np.array(coord1)
    arr2 = np.array(coord2)

    expected = min(np.linalg.norm(arr1 - arr2), 2 * cost_insert)
    assert abs(result - expected) < epsilon


@pytest.mark.parametrize('hash1, hash2, expected_cost',
                         [('sdfs', 'woerj23o', 2.0),
                          ('asdf', 'asdf', 0.0)])
def test_dirac_hash(hash1, hash2, expected_cost):
    node0 = Node(0, LabelHash(hash1))
    node1 = Node(1, LabelHash(hash2))

    edit_cost = EditCostVector(1., 1., 1., 1., 'euclidean', use_wl_attr=True)
    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected_cost