# -*- coding: utf-8 -*-

import pytest

from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.node import Node


@pytest.fixture()
def my_node():
    my_node = Node(1, LabelNodeLetter(1., 2.))
    my_node.label.get_attributes()
    return my_node


def test_simple_node(my_node):
    assert my_node.idx == 1
    assert my_node.label.get_attributes() == (1., 2.)


@pytest.mark.parametrize('in_args, expected',
                         [(Node(1, LabelNodeLetter(1., 2.)), True),
                          (Node(3, LabelNodeLetter(1., 2.)), False)])
def test_equality_node(my_node, in_args, expected):
    equality = my_node == in_args
    assert equality == expected


