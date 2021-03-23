import os
from glob import glob

from xmltodict import parse

from graph_pkg.utils.constants cimport EXTENSION_GRAPHS

cdef class LoaderBase:
    """
    Base class to load the datasets.
    Load and construct the graphs.

    Methods
    -------
    load()
    """

    def __init__(self, str folder):
        """

        :param folder:
        """
        self._folder = folder

    cpdef int _format_idx(self, str idx):
        raise NotImplementedError

    cpdef LabelBase _formatted_lbl_node(self, attr):
        raise NotImplementedError

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        raise NotImplementedError

    cpdef list load(self):
        """
        Load and construct the graphs.
        
        :return: list of constructed graphs
        """
        cdef object parsed_data

        files = os.path.join(self._folder, EXTENSION_GRAPHS)
        graph_files = glob(files)

        if not graph_files:
            raise FileNotFoundError(f'No graphs found in {self._folder}')

        graphs = []
        print('** Loading Graphs **')
        for graph_file in sorted(graph_files):
            with open(graph_file) as file:
                graph_text = "".join(file.readlines())
                *_, graph_filename = graph_file.split('/')
            parsed_data = parse(graph_text)

            self._construct_graph(graph_filename, parsed_data)

            graphs.append(self._constructed_graph)
            # break

        print(f'==> {len(graphs)} graphs loaded')
        return graphs

    cpdef void _construct_graph(self, str graph_filename, object parsed_data):
        graph_dict = parsed_data['gxl']['graph']

        graph_idx = graph_dict['@id']
        graph_edge_mode = graph_dict['@edgemode']
        nodes = graph_dict['node']
        edges = graph_dict['edge'] if 'edge' in graph_dict.keys() else []
        num_nodes = len(nodes)
        self._constructed_graph = Graph(graph_idx, graph_filename, num_nodes)

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
