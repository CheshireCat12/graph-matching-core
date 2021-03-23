import pytest
import numpy as np
import networkx as nx
from hierarchical_graph.centrality_measure.betweenness import Betweenness
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.node import Node
from graph_pkg.graph.edge import Edge
from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter


def test_betweenness_by_hand():
    graph = Graph('gr', 'gr.xml', 4)
    graph.add_node(Node(0, LabelNodeLetter(0, 0)))
    graph.add_node(Node(1, LabelNodeLetter(0, 0)))
    graph.add_node(Node(2, LabelNodeLetter(0, 0)))
    graph.add_node(Node(3, LabelNodeLetter(0, 0)))

    graph.add_edge(Edge(0, 1, LabelEdge(0)))
    graph.add_edge(Edge(1, 2, LabelEdge(0)))
    graph.add_edge(Edge(2, 3, LabelEdge(0)))

    betweenness = Betweenness()
    results = betweenness.calc_centrality_score(graph)
    results = np.asarray(results)

    graph2 = nx.Graph()
    graph2.add_node(1)
    graph2.add_node(2)
    graph2.add_node(3)
    graph2.add_node(4)
    graph2.add_edge(1, 2)
    graph2.add_edge(2, 3)
    graph2.add_edge(3, 4)

    expected_dict = nx.betweenness_centrality(graph2, normalized=False)
    expected = np.array([val for _, val in expected_dict.items()])
    print(results)

    assert np.linalg.norm(results - expected) < 1e-6
    # assert False

def test_betweenness_by_hand_big():
    graph = Graph('gr', 'gr.xml', 6)
    graph2 = nx.Graph()

    for i in range(6):
        graph.add_node(Node(i, LabelNodeLetter(0, 0)))
        graph2.add_node(i)

    ### Add edge to graph
    graph.add_edge(Edge(0, 1, LabelEdge(0)))
    graph.add_edge(Edge(0, 2, LabelEdge(0)))
    graph.add_edge(Edge(0, 4, LabelEdge(0)))
    graph.add_edge(Edge(0, 5, LabelEdge(0)))

    graph.add_edge(Edge(1, 2, LabelEdge(0)))

    graph.add_edge(Edge(2, 3, LabelEdge(0)))
    graph.add_edge(Edge(2, 5, LabelEdge(0)))

    graph.add_edge(Edge(3, 4, LabelEdge(0)))
    graph.add_edge(Edge(3, 5, LabelEdge(0)))

    betweenness = Betweenness()
    results = betweenness.calc_centrality_score(graph)
    results = np.asarray(results)

    ### Add edge to nx.graph
    graph2.add_edge(0, 1)
    graph2.add_edge(0, 2)
    graph2.add_edge(0, 4)
    graph2.add_edge(0, 5)

    graph2.add_edge(1, 2)

    graph2.add_edge(2, 3)
    graph2.add_edge(2, 5)

    graph2.add_edge(3, 4)
    graph2.add_edge(3, 5)

    expected_dict = nx.betweenness_centrality(graph2, normalized=False)
    expected = np.array([val for _, val in expected_dict.items()])
    print(results)

    assert np.linalg.norm(results - expected) < 1e-6
