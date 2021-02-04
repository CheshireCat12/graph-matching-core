import pytest

from graph_pkg.graph.graph import Graph
from graph_pkg.graph.node import Node
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter

@pytest.fixture()
def my_graph():
    return Graph('gr1', 1)

def test_simple_graph():
    my_graph = Graph('gr1', 1)

    assert my_graph.name == 'gr1'
    assert len(my_graph) == 1


@pytest.mark.parametrize('num_nodes',
                         [1, 5, 10])
def test_add_node(num_nodes):
    my_graph = Graph(f'gr{num_nodes}', num_nodes)
    nodes = []

    for i in range(num_nodes):
        tmp_node = Node(i, LabelNodeLetter(1, 1))
        nodes.append(tmp_node)
        my_graph.add_node(tmp_node)

    assert my_graph.get_nodes() == nodes


@pytest.mark.parametrize('num_nodes, error_idx',
                         [(5, 5),
                          (5, 8),])
def test_add_node_higher_than_num_nodes(num_nodes, error_idx):

    my_graph = Graph(f'gr{num_nodes}', num_nodes)
    tmp_node = Node(error_idx, LabelNodeLetter(1, 1))

    with pytest.raises(AssertionError) as execinfo:
        my_graph.add_node(tmp_node)

    error_msg = execinfo.value.args[0]
    expected_error_msg = f'The idx of the node {error_idx} exceed the number of nodes {num_nodes} authorized!'
    assert error_msg == expected_error_msg

