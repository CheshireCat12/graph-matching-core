# -*- coding: utf-8 -*-
"""
Test labels
"""
import pytest

from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.graph.label.label_node_AIDS import LabelNodeAIDS
from graph_pkg.graph.label.label_node_mutagenicity import LabelNodeMutagenicity
from graph_pkg.graph.label.label_node_NCI1 import LabelNodeNCI1


@pytest.mark.parametrize('in_args, expected',
                         [(5, 5),
                          (1., 1),
                          (0, 0)])
def test_label_edge(in_args, expected):
    label = LabelEdge(in_args)

    assert label.get_attributes() == (expected, )


@pytest.mark.parametrize('in_args',
                         [(5., 6.),
                          (1, 5.),
                          (43., 4)])
def test_label_letter(in_args):
    label = LabelNodeLetter(*in_args)

    assert label.get_attributes() == in_args

def test_label_AIDS():
    expected = ('C', 0, 1, 6., 5.)
    label = LabelNodeAIDS(*expected)

    assert label.get_attributes() == expected

def test_label_NCI1():
    expected = (3, )
    label = LabelNodeNCI1(*expected)

    assert label.get_attributes() == expected


@pytest.mark.parametrize('in_args',
                         [(5., 6.),
                          (1, 5.),
                          (43., 4)])
def test_label_lettre_to_string(in_args):
    label = LabelNodeLetter(*in_args)

    assert str(label) == f'Label attributes: {", ".join(str(float(element)) for element in in_args)}'


@pytest.mark.parametrize('lbl_1, lbl_2, expected',
                         [(LabelNodeLetter(1, 2), LabelNodeLetter(1, 2), True),
                          (LabelNodeLetter(1., 2.), LabelEdge(0), False),
                          (LabelEdge(0), LabelEdge(0), True),
                          (LabelEdge(12), LabelEdge(2), False),
                          (LabelNodeMutagenicity('C'), LabelNodeMutagenicity('C'), True),
                          (LabelNodeMutagenicity('Cl'), LabelNodeMutagenicity('O'), False),
                          (LabelNodeNCI1(3), LabelNodeNCI1(1), False),
                          (LabelNodeNCI1(2), LabelNodeNCI1(2), True)
                          ])
def test_label_equality(lbl_1, lbl_2, expected):
    equality = lbl_1 == lbl_2

    assert equality == expected

