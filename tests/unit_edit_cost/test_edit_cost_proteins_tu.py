import pytest

from graph_pkg.edit_cost.edit_cost_proteins_tu import EditCostProteinsTU
from graph_pkg.graph.label.label_node_proteins_tu import LabelNodeProteinsTU
from graph_pkg.graph.node import Node


@pytest.mark.parametrize('coord1, e_cost, expected',
                         [
                             ((1,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((0,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((2,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((0,), (11., 1., 1., 1., 'dirac'), 11.),
                             ((0,), (1., 1.9, 1.9, 1.9, 'dirac'), 1.),
                         ])
def test_dirac_proteins_tu_add_node(coord1, e_cost, expected):
    node0 = Node(0, LabelNodeProteinsTU(*coord1))

    edit_cost = EditCostProteinsTU(*e_cost)

    result = edit_cost.cost_insert_node(node0)

    assert result == expected


@pytest.mark.parametrize('coord1, e_cost, expected',
                         [
                             ((1,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((0,), (1., 1., 1., 1., 'dirac'), 1.),
                             ((1,), (16., 12., 18., 17., 'dirac'), 12.),
                         ])
def test_dirac_proteins_tu_delete_node(coord1, e_cost, expected):
    node0 = Node(0, LabelNodeProteinsTU(*coord1))

    edit_cost = EditCostProteinsTU(*e_cost)

    result = edit_cost.cost_delete_node(node0)

    assert result == expected


@pytest.mark.parametrize('coord1, coord2, e_cost, expected',
                         [
                          ((1,), (1,), (1., 1., 1., 1., 'dirac'), 0.),
                          ((0,), (1,), (1., 1., 1., 1., 'dirac'), 2.),
                          ((1,), (0,), (1., 1., 1., 1., 'dirac'), 2.),
                          ((1,), (2,), (3., 2., 2.5, 1., 'dirac'), 5.),

                         ])
def test_dirac_proteins_tu_substitution(coord1, coord2, e_cost, expected):
    node0 = Node(0, LabelNodeProteinsTU(*coord1))
    node1 = Node(1, LabelNodeProteinsTU(*coord2))

    edit_cost = EditCostProteinsTU(*e_cost)

    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected
