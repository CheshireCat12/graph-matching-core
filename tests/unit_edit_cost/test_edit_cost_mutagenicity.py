import pytest

from graph_pkg.edit_cost.edit_cost_mutagenicity import EditCostMutagenicity
from graph_pkg.graph.label.label_node_mutagenicity import LabelNodeMutagenicity
from graph_pkg.graph.node import Node


@pytest.mark.parametrize('coord1, coord2, e_cost, expected',
                         [(('C',), ('C',), (11., 11., 1.1, 1.1, 'dirac'), 0.),
                          (('C',), ('C',), (11., 11., 1.1, 1.1, 'dirac'), 0.),
                          (('H',), ('C',), (11., 11., 1.1, 1.1, 'dirac'), 22.),
                          (('H',), ('O',), (11., 11., 1.1, 1.1, 'dirac'), 22.),
                          (('Cl',), ('O',), (11., 11., 1.1, 1.1, 'dirac'), 22.),
                          (('Cl',), ('Cl',), (11., 11., 1.1, 1.1, 'dirac'), 0.),
                          (('Cl',), ('Na',), (11., 11., 1.1, 1.1, 'dirac'), 22.),

                          ])
def test_dirac_mutagenicity_norm(coord1, coord2, e_cost, expected):
    node0 = Node(0, LabelNodeMutagenicity(*coord1))
    node1 = Node(1, LabelNodeMutagenicity(*coord2))

    edit_cost = EditCostMutagenicity(*e_cost)
    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected
