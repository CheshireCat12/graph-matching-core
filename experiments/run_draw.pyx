""" Graph Drawer
@author: Anthony Gillioz

This script runs the graph drawer with SigmaJS.
The parameter file ../configuration/configuration_draw.yml has to be correctly settled before to run this script.
The configuration file contains all the parameters to run and save the graphs into a .js.

The centrality score of the nodes is computed on the fly and directly added to the be drawn.
"""
import random

from graph_pkg.utils.coordinator.coordinator import Coordinator
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweenness import Betweenness
from hierarchical_graph.utils.sigma_js import SigmaJS


def run_draw(parameters):
    """
    Draw the graphs into .js format to be used with the SigmaJS library.
    Randomly select the given number of graphs to be drawn.

    :param parameters: All the parameters from the config file.
    :return:
    """
    random.seed(42)

    coordinator = Coordinator(**parameters.coordinator)
    graphs = coordinator.graphs

    if parameters.random_graph:
        random.shuffle(graphs)

    selected_graphs = graphs[:parameters.num_graphs]

    sigma_js = SigmaJS(parameters.coordinator['dataset'],
                       parameters.folder_results)
    for graph in selected_graphs:
        centrality_score = _get_centrality_score(graph,
                                                 parameters.centrality_measure)
        sigma_js.save_to_sigma_with_score(graph,
                                          centrality_score,
                                          parameters.centrality_measure,
                                          level=0)

    print('-- Graphs correctly generated')


def _get_centrality_score(graph, centrality_measure):

    if centrality_measure == 'pagerank':
        measure = PageRank()
    elif centrality_measure == 'betweenness':
        measure = Betweenness()
    else:
        raise ValueError(f'Centrality measure: {centrality_measure} not accepted!')

    return measure.calc_centrality_score(graph)
