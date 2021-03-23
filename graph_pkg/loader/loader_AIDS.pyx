cdef class LoaderAIDS(LoaderBase):
    """
    Class to format the idx, the node's label and the edge's label correctly
    """

    def __init__(self, str folder='./data/AIDS/data'):
        super().__init__(folder)

    cpdef int _format_idx(self, str idx):
        return int(idx[1:]) - 1

    cpdef LabelBase _formatted_lbl_node(self, attr):
        symbol, chem, charge, x, y = attr
        symbol = str(symbol['string'])
        chem = int(chem['int'])
        charge = int(charge['int'])
        x = float(x['float'])
        y = float(y['float'])

        lbl_letter = LabelNodeAIDS(symbol, chem, charge, x, y)

        return lbl_letter

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        attr = attr['attr']
        return LabelEdge(int(attr['int']))
