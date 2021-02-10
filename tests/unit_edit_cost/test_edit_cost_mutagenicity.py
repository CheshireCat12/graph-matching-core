import pytest

from graph_pkg.edit_cost.edit_cost_mutagenicity import EditCostMutagenicity
from graph_pkg.graph.node import Node
from graph_pkg.graph.label.label_node_mutagenicity import LabelNodeMutagenicity

@pytest.mark.parametrize('coord1, coord2, expected',
                         [(('C',), ('C',), 0.),
                          (('C',), ('C',), 0.),
                          (('H',), ('C',), 1.),
                          (('H',), ('O',), 1.)])
def test_euclidean_norm(coord1, coord2, expected):
    node0 = Node(0, LabelNodeMutagenicity(*coord1))
    node1 = Node(1, LabelNodeMutagenicity(*coord2))

    edit_cost = EditCostMutagenicity(1., 1., 1., 1., 'dirac')
    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected
