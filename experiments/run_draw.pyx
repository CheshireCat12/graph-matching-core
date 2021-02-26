from graph_pkg.utils.coordinator.coordinator import Coordinator
from hierarchical_graph.algorithm.pagerank import pagerank_power
from hierarchical_graph.algorithm.betweeness import betweeness
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
        return pagerank_power(graph.adjacency_matrix)
    elif centrality_measure == 'betweeness':
        return betweeness(graph)

    raise ValueError(f'Centrality measure: {centrality_measure} not accepted!')
