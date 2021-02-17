
cdef class LoaderMutagenicity(LoaderBase):

    def __init__(self, str folder='./data/Mutagenicity/data/'):
        super().__init__(folder)

    cpdef int _format_idx(self, str idx):
        return int(idx) - 1

    cpdef LabelBase _formatted_lbl_node(self, attr):
        lbl_mutagenicity = LabelNodeMutagenicity(attr['string'])
        return lbl_mutagenicity

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        attr = attr['attr']
        return LabelEdge(int(attr['int']))
