from graph_pkg.utils.coordinator.coordinator import Coordinator


def run_draw(parameters):

    coordinator = Coordinator(**parameters['coordinator'])
    graphs = coordinator.graphs
    print(graphs[0])
    print(graphs[0]._set_edge())
    print(graphs[0].graph_to_json())
