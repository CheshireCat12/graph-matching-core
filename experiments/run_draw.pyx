from graph_pkg.utils.coordinator.coordinator import Coordinator
from hierarchical_graph.algorithm.pagerank import pagerank_power
from hierarchical_graph.algorithm.betweeness import betweeness
from hierarchical_graph.utils.functions import graph_to_sigma_with_score
import os
import json
from pathlib import Path
import random


def run_draw(parameters):
    random.seed(42)

    coordinator = Coordinator(**parameters['coordinator'])
    graphs = coordinator.graphs

    if parameters.random_graph:
        random.shuffle(graphs)

    selected_graphs = graphs[:parameters.num_graphs]

    Path(parameters['folder_results']).mkdir(parents=True, exist_ok=True)

    for graph in selected_graphs:
        centrality_score = _get_centrality_score(graph,
                                                 parameters.centrality_measure)
        data = graph_to_sigma_with_score(graph, centrality_score)

        filename = os.path.join(parameters['folder_results'],
                                f'{parameters.centrality_measure}_{graph.name}.json')
        with open(filename, 'w') as fp:
            json.dump(data, fp)

    print('-- Graphs correctly generated')


def _get_centrality_score(graph, centrality_measure):
    if centrality_measure == 'pagerank':
        return pagerank_power(graph.adjacency_matrix)
    elif centrality_measure == 'betweeness':
        return betweeness(graph)

    raise ValueError(f'Centrality measure: {centrality_measure} not accepted!')
