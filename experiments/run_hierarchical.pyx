from graph_pkg.utils.coordinator.coordinator import Coordinator
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweeness import Betweeness
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

    selected_graphs = graphs[:parameters.num_graphs]

    measure = ['pagerank', 'betweeness']
    strategy = ['one_by_one', 'multiple_by_one']

    for m, s in product(measure, strategy):

        parameters.centrality_measure = m
        parameters.strategy = s

        if parameters.centrality_measure == 'pagerank':
            measure = PageRank()
        elif parameters.centrality_measure == 'betweeness':
            measure = Betweeness()


        sigma_js = SigmaJS(parameters.coordinator['dataset'],
                           parameters.folder_results,
                           save_html=parameters.save_to_html)

        hierarchical_graph = HierarchicalGraph(selected_graphs[1:], measure, sigma_js)
        hierarchical_graph.create_hierarchy_sigma(parameters.strategy)
