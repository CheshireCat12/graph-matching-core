import pytest

from graph_pkg.edit_cost.edit_cost_NCI1 import EditCostNCI1
from graph_pkg.graph.label.label_node_NCI1 import LabelNodeNCI1
from graph_pkg.graph.node import Node


@pytest.mark.parametrize('coord1, coord2, e_cost, expected',
                         [((3,), (3,), (11., 11., 1.1, 1.1, 'dirac'), 0.),
                          ((4,), (3,), (11., 11., 1.1, 1.1, 'dirac'), 22.),
                          ((4,), (1,), (11., 11., 1.1, 1.1, 'dirac'), 22.),
                          ((12,), (1,), (11., 11., 1.1, 1.1, 'dirac'), 22.),
                          ((12,), (12,), (11., 11., 1.1, 1.1, 'dirac'), 0.),
                          ((12,), (7,), (11., 11., 1.1, 1.1, 'dirac'), 22.),

                          ])
def test_dirac_nci1_norm(coord1, coord2, e_cost, expected):
    node0 = Node(0, LabelNodeNCI1(*coord1))
    node1 = Node(1, LabelNodeNCI1(*coord2))

    edit_cost = EditCostNCI1(*e_cost)
    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected
