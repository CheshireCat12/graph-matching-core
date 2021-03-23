from graph_pkg.utils.coordinator.coordinator import Coordinator
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweenness import Betweenness
from hierarchical_graph.utils.sigma_js import SigmaJS
from hierarchical_graph.hierarchical_graph import HierarchicalGraph
import random
from itertools import product



def run_hierarchical(parameters):
    random.seed(42)

    coordinator = Coordinator(**parameters.coordinator)
    graphs = coordinator.graphs

    if parameters.random_graph:
        random.shuffle(graphs)

    # biggest_graph = None
    # max_size = float('-inf')
    # for graph in graphs:
    #     if len(graph) > max_size:
    #         print(graph.name)
    #         biggest_graph = graph
    #         max_size = len(graph)
    #
    # print(biggest_graph)
    sorted_by_len = sorted(graphs, key=lambda x: -len(x))
    for graph in sorted_by_len[-10:]:
        print(graph.name)


    selected_graphs = sorted_by_len[:parameters.num_graphs]

    percentages = [1.0, 0.8, 0.6, 0.4, 0.2]
    measures = ['pagerank', 'betweeness']
    strategies = ['compute_once']

    for strategy, measure, percentage in product(strategies, measures, percentages):

        parameters.percentage = percentage
        parameters.centrality_measure = measure
        parameters.deletion_strategy = strategy

        if parameters.centrality_measure == 'pagerank':
            measure = PageRank()
        elif parameters.centrality_measure == 'betweeness':
            measure = Betweenness()


        sigma_js = SigmaJS(parameters.coordinator['dataset'],
                           parameters.folder_results,
                           save_html=parameters.save_to_html)

        hierarchical_graph = HierarchicalGraph(selected_graphs[1:], measure, sigma_js)
        # hierarchical_graph.create_hierarchy_sigma(parameters.strategy)
        hierarchical_graph.create_hierarchy_percent(selected_graphs,
                                                   percentage_remaining=percentage,
                                                   deletion_strategy=strategy,
                                                   verbose=False)

        # break