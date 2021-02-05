from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_AIDS import LabelNodeAIDS
from graph_pkg.loader.loader_base import LoaderBase


class LoaderAIDS(LoaderBase):
    _num_lines_to_trim_front = 1
    _num_lines_to_trim_end = None
    _num_chars_to_trim_start = 64
    _num_chars_to_trim_end = 7

    def __init__(self, folder):
        super().__init__(folder)

    def _format_idx(self, idx):
        return int(idx[1:]) - 1

    def _formated_lbl_node(self, attr):
        symbol, chem, charge, x, y = attr
        symbol = str(symbol['string'])
        chem = int(chem['int'])
        charge = int(charge['int'])
        x = float(x['float'])
        y = float(y['float'])

        lbl_letter = LabelNodeAIDS(symbol, chem, charge, x, y)

        return lbl_letter

    def _formated_lbl_edge(self, attr):
        attr = attr['attr']
        return LabelEdge(int(attr['int']))
