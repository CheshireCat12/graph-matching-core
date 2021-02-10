from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
from graph_pkg.loader.loader_base import LoaderBase
import os

class LoaderLetter(LoaderBase):

    _folder = './data/Letter/Letter/'

    def __init__(self, spec='LOW'):
        super().__init__()
        self._folder = os.path.join(self._folder, spec, '')

    def _format_idx(self, idx):
        return int(idx[1:])

    def _formatted_lbl_node(self, attr):
       data = [float(val['float']) for val in attr]
       lbl_letter = LabelNodeLetter(*data)

       return lbl_letter

    def _formatted_lbl_edge(self, attr):
        return LabelEdge(0)
