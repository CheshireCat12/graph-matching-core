# -*- coding: utf-8 -*-
"""
Test labels
"""
import pytest
import numpy as np
from graph_pkg_core.graph.label.label_edge import LabelEdge
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector


@pytest.mark.parametrize('in_args, expected',
                         [(5, 5),
                          (1., 1),
                          (0, 0)])
def test_label_edge(in_args, expected):
    label = LabelEdge(in_args)

    assert label.get_attributes() == (expected, )


@pytest.mark.parametrize('in_args',
                         [np.array([5., 6.]),
                          np.array([1, 5.]),
                          np.array([43., 4])])
def test_label_vector(in_args):
    label = LabelNodeVector(in_args)

    assert np.array_equal(label.get_attributes()[0], in_args) == True


@pytest.mark.parametrize('in_args',
                         [np.array([5., 6.]),
                          np.array([1, 5.]),
                          np.array([43., 4])])
def test_label_vector_to_string(in_args):
    label = LabelNodeVector(in_args)

    assert str(label) == f'Label attributes: {", ".join(str(element) for element in (in_args,))}'


@pytest.mark.parametrize('lbl_1, lbl_2, expected',
                         [(LabelNodeVector(np.array([1, 2])), LabelNodeVector(np.array([1, 2])), True),
                          (LabelNodeVector(np.array([1, 3])), LabelNodeVector(np.array([1, 2])), False),
                          (LabelEdge(0), LabelEdge(0), True),
                          (LabelEdge(12), LabelEdge(2), False),
                          ])
def test_label_equality(lbl_1, lbl_2, expected):
    equality = lbl_1 == lbl_2

    assert equality == expected

