# -*- coding: utf-8 -*-

import numpy as np
import pytest

from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.node import Node


@pytest.fixture()
def my_node():
    my_node = Node(1, LabelNodeVector(np.array([1., 2.])))

    return my_node


def test_simple_node(my_node):
    assert my_node.idx == 1
    assert np.array_equal(my_node.label.get_attributes()[0], np.array([1., 2.]))


@pytest.mark.parametrize('node1, node2, expected',
                         [(Node(1, LabelNodeVector(np.array([1., 2.]))),
                           Node(1, LabelNodeVector(np.array([1., 2.]))),
                           True),
                          (Node(1, LabelNodeVector(np.array([1., 2.]))),
                           Node(3, LabelNodeVector(np.array([1., 4.]))),
                           False),
                          (Node(1, LabelNodeVector(np.array([1., 2.]))),
                           Node(1, LabelNodeVector(np.array([1, 2.1]))),
                           False)])
def test_equality_node(node1, node2, expected):
    equality = node1 == node2

    assert equality == expected
