from graph_pkg.graph.label.label_edge import LabelEdge
from graph_pkg.graph.label.label_node_letter import LabelNodeLetter
import os

cdef class LoaderLetter(LoaderBase):

    # _folder = './data/Letter/Letter/'

    def __cinit__(self, str spec):
        super().__init__(os.path.join('./data/Letter/Letter/', spec, ''))
        folder = './data/Letter/Letter/'
        print(folder)
        folder = os.path.join(folder, spec, '')

        print(folder)

    cpdef int _format_idx(self, str idx):
        return int(idx[1:])

    cpdef LabelBase _formatted_lbl_node(self, attr):
       data = [float(val['float']) for val in attr]
       lbl_letter = LabelNodeLetter(*data)

       return lbl_letter

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        return LabelEdge(0)
