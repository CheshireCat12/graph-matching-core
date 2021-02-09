from xmltodict import parse
from glob import glob
from abc import ABC, abstractmethod

from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.node import Node

# from main import Graph, Node, Edge


class LoaderBase(ABC):

    _num_lines_to_trim_front: int = NotImplemented
    _num_lines_to_trim_end: int = NotImplemented
    _num_chars_to_trim_start: int = NotImplemented
    _num_chars_to_trim_end: int = NotImplemented

    _folder: str = NotImplemented

    __EXTENSION = '.gxl'

    def __init__(self):
        self._current_graph_text = None
        self._parsed_data = None
        self._constructed_graph = None

    def _clean_text(self):
        start = self._num_lines_to_trim_front
        end = None
        if self._num_lines_to_trim_end is not None:
            end = -self._num_lines_to_trim_end

        removed_front_and_end_lines = self._current_graph_text[start: end]

        start = self._num_chars_to_trim_start
        end = None
        if self._num_chars_to_trim_end is not None:
            end = -self._num_chars_to_trim_end
        joined_xml_without_end_chars = "".join(removed_front_and_end_lines)[start:end]
        self._current_graph_text = joined_xml_without_end_chars

    @abstractmethod
    def _format_idx(self, idx):
        pass

    @abstractmethod
    def _formated_lbl_node(self, attr):
        pass

    @abstractmethod
    def _formated_lbl_edge(self, attr):
        pass

    def _construct_graph(self):
        graph_dict = self._parsed_data['graph']

        graph_name = graph_dict['@id']
        num_nodes = len(graph_dict['node'])
        self._constructed_graph = Graph(graph_name, num_nodes)

        # print(f'Construct Graph: {graph_name}')

        # variable used to check if there is no gap in the indexes from the xml files
        idx_verification = 0

        if not isinstance(graph_dict['node'], list):
            graph_dict['node'] = [graph_dict['node']]
        for element in graph_dict['node']:
            idx = self._format_idx(element['@id'])

            assert idx == idx_verification, f'There is a gap in the index {idx} from {graph_name}'

            lbl_node = self._formated_lbl_node(element['attr'])
            self._constructed_graph.add_node(Node(idx, lbl_node))

            idx_verification += 1

        if 'edge' not in graph_dict.keys():
            return
        if not isinstance(graph_dict['edge'], list):
            graph_dict['edge'] = [graph_dict['edge']]
        for element in graph_dict['edge']:
            idx_from = self._format_idx(element['@from'])
            idx_to = self._format_idx(element['@to'])
            lbl_edge = self._formated_lbl_edge(element)
            tmp_edge = Edge(idx_from, idx_to, lbl_edge)

            self._constructed_graph.add_edge(tmp_edge)

    def load(self):
        graph_files = glob(f'{self._folder}*{self.__EXTENSION}')

        graphs = []
        print('** Loading Graphs **')
        for graph_file in graph_files:
            with open(graph_file) as file:
                self._current_graph_text = file.readlines()
                self._clean_text()
                self._parsed_data = parse(self._current_graph_text)
                self._construct_graph()

            graphs.append(self._constructed_graph)
            # break

        return graphs
