from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.loader.loader_base import LoaderBase
import os

class LoaderLetter(LoaderBase):
    _num_lines_to_trim_front = 3
    _num_lines_to_trim_end = None
    _num_chars_to_trim_start = 0
    _num_chars_to_trim_end = 6

    _folder = './data/Letter/Letter/'

    def __init__(self, spec='LOW'):
        super().__init__()
        self._folder = os.path.join(self._folder, spec, '')

    def _format_idx(self, idx):
        return int(idx[1:])

    def _formated_lbl_node(self, attr):
       data = [float(val['float']) for val in attr]
       lbl_letter = LabelNodeLetter(*data)

       return lbl_letter

    def _formated_lbl_edge(self, attr):
        return LabelEdge(0)
