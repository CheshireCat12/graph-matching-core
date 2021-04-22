import pytest
from graph_pkg.graph.node import Node

from graph_pkg.edit_cost.edit_cost_protein import EditCostProtein
from graph_pkg.graph.label.label_node_protein import LabelNodeProtein
from graph_pkg.algorithm.levenshtein_distance import LevenshteinDistance

levi = LevenshteinDistance()

@pytest.mark.parametrize('coord1, coord2, e_cost, expected',
                         [((0, 9, 'DLLAELQWR'), (0, 9, 'EDGLRKLLNE'), (11, 11, 1., 1., 'sed', 1., 1., 1.), levi.compute_string_edit_distance_normalized_cpd("DLLAELQWR", "EDGLRKLLNE")),
                          ((0, 12, 'LAPILTMRRFQQ'), (1, 4, 'KIKN'), (11, 11, 1., 1., 'sed', 1., 1., 1.), 22.),
                          ((1, 9, 'RPIALV'), (1, 9, 'TLYCGF'), (11, 11, 1., 1., 'sed', 1., 1., 1.), levi.compute_string_edit_distance_normalized_cpd('RPIALV','TLYCGF')),
                          ((1, 9, 'RPIALV'), (0, 9, 'PYEFYQFWI'), (1., 1., 1., 1., 'sed', 1., 1., 1.), 2.),
                          ])
def test_dirac_protein_norm(coord1, coord2, e_cost, expected):
    node0 = Node(0, LabelNodeProtein(*coord1))
    node1 = Node(1, LabelNodeProtein(*coord2))

    edit_cost = EditCostProtein(*e_cost)
    result = edit_cost.cost_substitute_node(node0, node1)

    assert result == expected
    assert edit_cost.cost_insert_node(node0) == e_cost[0]
    assert edit_cost.cost_delete_node(node1) == e_cost[1]
