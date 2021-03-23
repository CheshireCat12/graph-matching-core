import random

from graph_pkg.utils.coordinator.coordinator import Coordinator
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweenness import Betweenness
from hierarchical_graph.utils.sigma_js import SigmaJS
from hierarchical_graph.hierarchical_graphs import HierarchicalGraphs


__MEASURES = {'pagerank': PageRank(),
              'betweenness': Betweenness()}

def run_hierarchical(parameters):
    random.seed(42)

    coordinator = Coordinator(**parameters.coordinator)
    graphs = coordinator.graphs

    if parameters.random_graph:
        random.shuffle(graphs)

    # Select the 10 biggest graphs
    sorted_by_len = sorted(graphs, key=lambda x: -len(x))
    for graph in sorted_by_len[:10]:
        print(graph.name)

    selected_graphs = sorted_by_len[:parameters.num_graphs]

    measures = ['pagerank', 'betweenness']

    sigma_js = SigmaJS(parameters.coordinator['dataset'],
                       parameters.folder_results,
                       save_html=parameters.save_to_html)

    for measure in measures:
        h_graphs = HierarchicalGraphs(selected_graphs, __MEASURES[measure])
        _save_h_graphs_to_js(sigma_js, h_graphs, __MEASURES[measure])



def _save_h_graphs_to_js(sigma_js, h_graphs, measure):
    original_graphs = h_graphs.hierarchy[1.0]

    for level, (percentage, graphs) in enumerate(h_graphs.hierarchy.items()):

        for idx, graph in enumerate(graphs):
            original_size = len(original_graphs[idx])
            current_size = len(graph)
            extra_info = f'percentage_{percentage}'

            extra_info_nodes = f'Current nodes/Total nodes: {current_size}/{original_size} <br>' \
                               f'Percentage remaining: {percentage * 100:.0f}%'

            centrality_score = measure.calc_centrality_score(graph)

            sigma_js.save_to_sigma_with_score(graph,
                                                   centrality_score,
                                                   measure.name,
                                                   level=level,
                                                   extra_info=extra_info,
                                                   extra_info_nodes=extra_info_nodes)