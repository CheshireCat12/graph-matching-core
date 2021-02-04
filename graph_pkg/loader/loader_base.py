from xmltodict import parse
from glob import glob
from abc import ABC, abstractmethod


# from main import Graph, Node, Edge


class LoaderBase(ABC):

    _num_lines_to_trim_front: int = NotImplemented
    _num_lines_to_trim_end: int = NotImplemented
    _num_chars_to_trim_end: int = NotImplemented

    __EXTENSION = '.gxl'

    def __init__(self, folder):
        self.folder = folder
        self._current_graph_text = None
        self._parsed_data = None
        self._constructed_graph = None

    def _clean_text(self):
        print('+++')
        start = self._num_lines_to_trim_front
        end = None
        if self._num_lines_to_trim_end is not None:
            end = -self._num_lines_to_trim_end

        removed_front_and_end_lines = self._current_graph_text[start: end]
        print(removed_front_and_end_lines)

        end = None
        if self._num_chars_to_trim_end is not None:
            end = -self._num_chars_to_trim_end
        joined_xml_without_end_chars = "".join(removed_front_and_end_lines)[:end]
        self._current_graph_text = joined_xml_without_end_chars

    @abstractmethod
    def _format_xml(self):
        pass

    def load(self):
        graph_files = glob(f'{self.folder}*{self.__EXTENSION}')
        graphs = []

        for graph_file in graph_files:
            with open(graph_file) as file:
                self._current_graph_text = file.readlines()
                print(self._current_graph_text)
                self._clean_text()
                print(self._current_graph_text)
                self._parsed_data = parse(self._current_graph_text)
                self._format_xml()

            graphs.append(self._constructed_graph)
            break

        return graphs
