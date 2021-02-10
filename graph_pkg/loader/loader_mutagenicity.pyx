from graph_pkg.loader.loader_base import LoaderBase
from graph_pkg.graph.label.label_node_mutagenicity import LabelNodeMutagenicity
from graph_pkg.graph.label.label_edge import LabelEdge

class LoaderMutagenicity(LoaderBase):

    _folder = './data/Mutagenicity/data/'

    def __init__(self):
        super().__init__()

    def _format_idx(self, idx):
        return int(idx) - 1

    def _formatted_lbl_node(self, attr):
        lbl_mutagenicity = LabelNodeMutagenicity(attr['string'])
        return lbl_mutagenicity

    def _formatted_lbl_edge(self, attr):
        attr = attr['attr']
        return LabelEdge(int(attr['int']))
