# -*- coding: utf-8 -*-

from graph_pkg.graph.network import Node

my_node = Node(1, 'C')


def test_idx_node():
    assert my_node.get_id() == 1


def test_label_node():
    assert my_node.get_label() == 'C'