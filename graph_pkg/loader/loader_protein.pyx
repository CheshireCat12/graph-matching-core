from graph_pkg.graph.label.label_edge cimport LabelEdge
from graph_pkg.graph.label.label_node_protein cimport LabelNodeProtein

cdef class LoaderProtein(LoaderBase):

    def __init__(self, str folder='./data/Protein/data/'):
        super().__init__(folder)

    cpdef int _format_idx(self, str idx):
        return int(idx) - 1

    cpdef LabelBase _formatted_lbl_node(self, attr):
        type_, aa_length, sequence = attr
        type_ = int(type_['int'])
        aa_length = int(aa_length['int'])
        sequence = str(sequence['int'])
        return LabelNodeProtein(type_, aa_length, sequence)

    cpdef LabelBase _formatted_lbl_edge(self, attr):
        return LabelEdge(0)
