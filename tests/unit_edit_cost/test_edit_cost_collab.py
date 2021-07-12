import pytest

from graph_pkg.edit_cost.edit_cost_collab import EditCostCollab
from graph_pkg.graph.label.label_node_collab import LabelNodeCollab
from graph_pkg.graph.node import Node


@pytest.mark.parametrize('coord1, e_cost, expected',
                         [
                             ((1,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((0,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((2,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((0,), (11., 1., 1., 1., 'dirac'), 11.),
                             ((0,), (1., 1.9, 1.9, 1.9, 'dirac'), 1.),
                         ])
def test_dirac_collab_add_node(coord1, e_cost, expected):
    node0 = Node(0, LabelNodeCollab(*coord1))

    edit_cost = EditCostCollab(*e_cost)

    result = edit_cost.cost_insert_node(node0)

    assert result == expected


@pytest.mark.parametrize('coord1, e_cost, expected',
                         [
                             ((1,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((1,), (1., 1.5, 1., 1., 'dirac'), 1.5),
                             ((1,), (16., 12., 18., 17., 'dirac'), 12.),
                         ])
def test_dirac_collab_delete_node(coord1, e_cost, expected):
    node0 = Node(0, LabelNodeCollab(*coord1))

    edit_cost = EditCostCollab(*e_cost)

    result = edit_cost.cost_delete_node(node0)

    assert result == expected


@pytest.mark.parametrize('coord1, coord2, e_cost, expected',
                         [
                             ((1,), (1,), (1., 1., 1., 1., 'dirac'), 0.),
                             ((1,), (1,), (1., 1., 1., 1., 'dirac'), 0.),
                             ((1,), (1,), (1., 1., 1., 1., 'dirac'), 0.),
                             ((1,), (1,), (3., 2., 2.5, 1., 'dirac'), 0.),

                         ])
def test_dirac_collab_substitution(coord1, coord2, e_cost, expected):
    node0 = Node(0, LabelNodeCollab(*coord1))
    node1 = Node(1, LabelNodeCollab(*coord2))

    edit_cost = EditCostCollab(*e_cost)

    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected