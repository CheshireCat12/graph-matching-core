
cdef class LoaderMutagenicity(LoaderBase):

    def __cinit__(self):
        folder = './data/Mutagenicity/data/'
        self._init_folder(folder)

    cpdef int _format_idx(self, str idx):

        return int(idx) - 1

    cpdef LabelBase _formatted_lbl_node(self, attr):
        lbl_mutagenicity = LabelNodeMutagenicity(attr['string'])
        return lbl_mutagenicity

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        attr = attr['attr']
        return LabelEdge(int(attr['int']))
