from graph_pkg.graph.graph cimport Graph
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node

def _round(num):
    return round(num, 3)


def graph_to_sigma_with_score(Graph graph, double[::1] centrality_score):
    cdef:
        Edge edge
        Node node

    graph_data = {'nodes': [], 'edges': []}

    for node, score in zip(graph.nodes, centrality_score):
        score = _round(score)
        lbl = f'PageRank: {score}; {node.label.sigma_attributes()}'
        x, y = [_round(val) for val in node.label.sigma_position()]
        graph_data['nodes'].append({
            'id': f'{node.idx}',
            'label': lbl,
            'x': x,
            'y': y,
            'size': score
        })

    for idx, edge in enumerate(graph._set_edge()):
        graph_data['edges'].append({
            'id': f'{idx}',
            'source': f'{edge.idx_node_start}',
            'target': f'{edge.idx_node_end}',
            'label': f'{edge.weight.valence}'
        })

    return graph_data