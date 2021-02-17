import os

cdef class LoaderLetter(LoaderBase):

    def __init__(self, str folder='./data/Letter/Letter/HIGH'):
        super().__init__(folder)
        # self._init_folder(folder)

    cpdef int _format_idx(self, str idx):
        return int(idx[1:])

    cpdef LabelBase _formatted_lbl_node(self, attr):
       data = [float(val['float']) for val in attr]
       lbl_letter = LabelNodeLetter(*data)

       return lbl_letter

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        return LabelEdge(0)
