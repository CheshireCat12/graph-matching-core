cdef class LoaderNCI1(LoaderBase):

    def __init__(self, str folder='./data/NCI1/data/'):
        super().__init__(folder)

    cpdef int _format_idx(self, str idx):
        return int(idx)

    cpdef LabelBase _formatted_lbl_node(self, attr):
        lbl_NCI1 = LabelNodeNCI1(int(attr['int']))
        return lbl_NCI1

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        return LabelEdge(0)
