import pytest

from graph_pkg.edit_cost.edit_cost_letter_minkowski import EditCostLetterMinkowski
from graph_pkg.graph.node import Node
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter

def test_euclidean_norm():
    node0 = Node(0, LabelNodeLetter(0., 0.))
    node1 = Node(1, LabelNodeLetter(3., 4.))

    edit_cost = EditCostLetterMinkowski(2)
    result = edit_cost.cost_substitute_node(node0, node1)
    assert result == 5