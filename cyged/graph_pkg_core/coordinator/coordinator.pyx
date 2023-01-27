from typing import List, Tuple

import networkx as nx
import numpy as np

from cyged.graph_pkg_core.algorithm.graph_edit_distance cimport GED
from cyged.graph_pkg_core.edit_cost.edit_cost_vector cimport EditCostVector
from cyged.graph_pkg_core.graph.edge cimport Edge
from cyged.graph_pkg_core.graph.graph cimport Graph
from cyged.graph_pkg_core.graph.label.label_edge cimport LabelEdge
from cyged.graph_pkg_core.graph.label.label_node_vector cimport LabelNodeVector
from cyged.graph_pkg_core.graph.node cimport Node

def _convert_graph(idx_graph: int,
                   graph: nx.Graph) -> Graph:
    """
    Convert the given nx.Graph into own graph format.

    Args:
        idx_graph:
        graph:

    Returns:

    """
    new_graph = Graph(name=str(idx_graph),
                      filename=f'gr_{idx_graph}.graphml',
                      num_nodes=len(graph.nodes))
    node_attr = 'x'

    for idx_node, node_data in graph.nodes(data=True):
        lbl_node = LabelNodeVector(node_data[node_attr])
        node = Node(int(idx_node), lbl_node)

        new_graph.add_node(node)

    for idx_start, idx_stop in graph.edges:
        edge = Edge(int(idx_start), int(idx_stop), LabelEdge(0))

        new_graph.add_edge(edge)

    return new_graph


class Coordinator:

    def __init__(self,
                 parameters_edit_cost: Tuple,
                 graphs: List[nx.Graph],
                 classes: np.ndarray):
        """

        Args:
            parameters_edit_cost:
            root_dataset:
        """
        self.parameters_edit_cost = parameters_edit_cost
        self.edit_cost = EditCostVector(*self.parameters_edit_cost)
        self.ged = GED(self.edit_cost)
        self.graphs = [_convert_graph(idx_graph, graph)
                       for idx_graph, graph in enumerate(graphs)]
        self.classes = classes
