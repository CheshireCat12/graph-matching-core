from loader_base import LoaderBase


class LoaderLetter(LoaderBase):

    _num_lines_to_trim_front = 3
    _num_lines_to_trim_end = None
    _num_chars_to_trim_end = 6

    def __init__(self, folder):
        super().__init__(folder)

    def _format_xml(self):
        graph_dict = self._parsed_data['graph']
        print(graph_dict['@id'])
        self._constructed_graph = Graph(graph_dict['@id'])

        print('Insert Vertices')
        if not isinstance(graph_dict['node'], list):
            graph_dict['node'] = [graph_dict['node']]
        for element in graph_dict['node']:
            idx = element['@id']
            data = [val['float'] for val in element['attr']]
            self._constructed_graph.add_node(Node(idx, data))

        print('Insert Edges:')
        if 'edge' not in graph_dict.keys():
            return
        if not isinstance(graph_dict['edge'], list):
            graph_dict['edge'] = [graph_dict['edge']]
        for element in graph_dict['edge']:
            print(element['@from'])
            print(element['@from'], element['@to'])
            self._constructed_graph.add_edge(Edge(element['@from'], element['@to']))