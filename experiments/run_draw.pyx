from graph_pkg.utils.coordinator.coordinator import Coordinator
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweeness import Betweeness
from hierarchical_graph.utils.sigma_js import SigmaJS
import os
import json
from pathlib import Path
import random


def run_draw(parameters):
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
                                          parameters.centrality_measure, level=0)

    print('-- Graphs correctly generated')


def _get_centrality_score(graph, centrality_measure):

    if centrality_measure == 'pagerank':
        measure = PageRank()
    elif centrality_measure == 'betweeness':
        measure = Betweeness()
    else:
        raise ValueError(f'Centrality measure: {centrality_measure} not accepted!')

    return measure.calc_centrality_score(graph)
