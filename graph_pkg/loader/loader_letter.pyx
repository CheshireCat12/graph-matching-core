import os

cdef class LoaderLetter(LoaderBase):

    def __cinit__(self, str spec):
        folder = './data/Letter/Letter/'
        folder_spec = os.path.join(folder, spec, '')
        self._init_folder(folder_spec)

    cpdef int _format_idx(self, str idx):
        return int(idx[1:])

    cpdef LabelBase _formatted_lbl_node(self, attr):
       data = [float(val['float']) for val in attr]
       lbl_letter = LabelNodeLetter(*data)

       return lbl_letter

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        return LabelEdge(0)
