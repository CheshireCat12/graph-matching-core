# -*- coding: utf-8 -*-
"""
Test the label letter
"""
import pytest

from graph_pkg.graph.label.label_node.label_node_letter import LabelNodeLetter
from graph_pkg.graph.label.label_node.label_node_AIDS import LabelNodeAIDS

@pytest.mark.parametrize('in_args',
                         [(5., 6.),
                          (1, 5.),
                          (43., 4)])
def test_label_letter(in_args):
    label = LabelNodeLetter(*in_args)

    assert label.get_attributes() == in_args

@pytest.mark.parametrize('in_args',
                         [(5., 6.),
                          (1, 5.),
                          (43., 4)])
def test_label_lettre_to_string(in_args):
    label = LabelNodeLetter(*in_args)

    assert str(label) == f'Label attributes: {", ".join(str(float(element)) for element in in_args)}'


def test_label_AIDS():
    expected = ('C', 0, 1, 6., 5.)
    label = LabelNodeAIDS(*expected)

    with pytest.raises(NotImplementedError) as execinfo:
        label.get_attributes()
