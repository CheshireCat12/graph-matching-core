from abc import ABC, abstractmethod
from glob import glob
from xmltodict import parse

from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.node import Node


class LoaderBase(ABC):

    _folder: str = NotImplemented
    __EXTENSION = '.gxl'

    def __init__(self):
        # self._current_graph_text = None
        self._parsed_data = None
        self._constructed_graph = None

    @abstractmethod
    def _format_idx(self, idx):
        pass

    @abstractmethod
    def _formatted_lbl_node(self, attr):
        pass

    @abstractmethod
    def _formatted_lbl_edge(self, attr):
        pass

    def _construct_graph(self):
        graph_dict = self._parsed_data['gxl']['graph']

        graph_idx = graph_dict['@id']
        graph_edge_mode = graph_dict['@edgemode']
        nodes = graph_dict['node']
        edges = graph_dict['edge'] if 'edge' in graph_dict.keys() else []
        num_nodes = len(nodes)
        self._constructed_graph = Graph(graph_idx, num_nodes)

        # variable used to check if there is no gap in the indexes from the xml files
        idx_verification = 0

        if not isinstance(nodes, list):
            nodes = [nodes]
        for element in nodes:
            idx = self._format_idx(element['@id'])

            assert idx == idx_verification, f'There is a gap in the index {idx} from {graph_idx}'

            lbl_node = self._formatted_lbl_node(element['attr'])
            self._constructed_graph.add_node(Node(idx, lbl_node))

            idx_verification += 1

        if not isinstance(edges, list):
            edges = [edges]
        for element in edges:
            idx_from = self._format_idx(element['@from'])
            idx_to = self._format_idx(element['@to'])
            lbl_edge = self._formatted_lbl_edge(element)
            tmp_edge = Edge(idx_from, idx_to, lbl_edge)

            self._constructed_graph.add_edge(tmp_edge)

    def load(self):
        graph_files = glob(f'{self._folder}*{self.__EXTENSION}')

        graphs = []
        print('** Loading Graphs **')
        for graph_file in sorted(graph_files):
            with open(graph_file) as file:
                graph_text = "".join(file.readlines())
            self._parsed_data = parse(graph_text)
            self._construct_graph()

            graphs.append(self._constructed_graph)
            # break

        print(f'==> {len(graphs)} graphs loaded')
        return graphs
