from abc import ABC, abstractmethod
from glob import glob
from xmltodict import parse

from graph_pkg.graph.edge import Edge
from graph_pkg.graph.graph import Graph
from graph_pkg.graph.node import Node


cdef class LoaderBase:

    # _folder = NotImplemented
    # __EXTENSION = '.gxl'

    def __cinit__(self, str folder):
        self._folder = folder
        self.__EXTENSION = '.gxl'

    cpdef int _format_idx(self, str idx):
        raise NotImplementedError

    cpdef LabelBase _formatted_lbl_node(self, attr):
        raise NotImplementedError

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        raise NotImplementedError

    cpdef list load(self):
        files = f'{self._folder}*{self.__EXTENSION}'
        graph_files = glob(files)
        print(files)

        print(graph_files)
        graphs = []
        print('** Loading Graphs **')
        for graph_file in sorted(graph_files):
            with open(graph_file) as file:
                graph_text = "".join(file.readlines())
            print(graph_text)
            self._parsed_data = parse(graph_text)
            print(self._parsed_data)
            self._construct_graph()

            graphs.append(self._constructed_graph)
            # break

        print(f'==> {len(graphs)} graphs loaded')
        return graphs

    cpdef void _construct_graph(self):
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
