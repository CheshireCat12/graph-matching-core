import pytest

from graph_pkg.edit_cost.edit_cost_AIDS import EditCostAIDS
from graph_pkg.graph.label.label_node_AIDS import LabelNodeAIDS
from graph_pkg.graph.node import Node


@pytest.mark.parametrize('coord1, coord2, e_cost, expected',
                         [(('C', 1, 1, 0., 0.), ('C', 1, 1, 3., 4.), (1.1, 1.1, 0.1, 0.1, 'dirac'), 0.),
                          (('C', 1, 3, 6., 0.3), ('C', 1, 1, 3., 4.), (1.1, 1.1, 0.1, 0.1, 'dirac'), 0.),
                          (('H', 1, 1, 0., 0.), ('C', 1, 1, 3., 4.), (1.1, 1.1, 0.1, 0.1, 'dirac'), 2.2),
                          (('H', 4, 5, 0.5, 4.), ('O', 1, 1, 3., 4.), (1.1, 1.1, 0.1, 0.1, 'dirac'), 2.2),
                          (('Cl', 1, 1, 0., 0.), ('C', 1, 1, 3., 4.), (1.1, 1.1, 0.1, 0.1, 'dirac'), 2.2),
                          (('Cl', 4, 5, 0.5, 4.), ('Cl', 1, 1, 3., 4.), (1.1, 1.1, 0.1, 0.1, 'dirac'), 0.),
                          (('O', 1, 1, 0., 0.), ('Si', 1, 1, 3., 4.), (1.1, 1.1, 0.1, 0.1, 'dirac'), 2.2),
                          (('Si', 4, 5, 0.5, 4.), ('Si', 1, 1, 3., 4.), (1.1, 1.1, 0.1, 0.1, 'dirac'), 0.)])
def test_dirac_aids_norm(coord1, coord2, e_cost, expected):
    node0 = Node(0, LabelNodeAIDS(*coord1))
    node1 = Node(1, LabelNodeAIDS(*coord2))

    edit_cost = EditCostAIDS(*e_cost)
    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected
    assert edit_cost.cost_insert_node(node0) == e_cost[0]
    assert edit_cost.cost_delete_node(node1) == e_cost[1]
