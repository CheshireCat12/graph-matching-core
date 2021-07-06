cdef class LoaderCollab(LoaderBase):
    """
    Class to format the idx, the node's label and the edge's label correctly
    """

    def __init__(self, str folder='./data/COLLAB/data/'):
        super().__init__(folder)

    cpdef int _format_idx(self, str idx):
        return int(idx)

    cpdef LabelBase _formatted_lbl_node(self, attr):
        lbl_collab = LabelNodeCollab(int(attr['int']))
        return lbl_collab

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        return LabelEdge(0)
