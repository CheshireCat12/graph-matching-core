# from graph_pkg.graph.label.label_edge import LabelEdge
# from graph_pkg.graph.label.label_node_AIDS import LabelNodeAIDS
# from graph_pkg.loader.loader_base import LoaderBase
#
#
# class LoaderAIDS(LoaderBase):
#
#     _folder = './data/AIDS/data/'
#
#     def __init__(self):
#         super().__init__()
#
#     def _format_idx(self, idx):
#         return int(idx[1:]) - 1
#
#     def _formatted_lbl_node(self, attr):
#         symbol, chem, charge, x, y = attr
#         symbol = str(symbol['string'])
#         chem = int(chem['int'])
#         charge = int(charge['int'])
#         x = float(x['float'])
#         y = float(y['float'])
#
#         lbl_letter = LabelNodeAIDS(symbol, chem, charge, x, y)
#
#         return lbl_letter
#
#     def _formatted_lbl_edge(self, attr):
#         attr = attr['attr']
#         return LabelEdge(int(attr['int']))
