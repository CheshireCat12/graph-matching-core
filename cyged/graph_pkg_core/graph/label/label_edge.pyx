from cyged.graph_pkg_core.graph.label.label_base cimport LabelBase


cdef class LabelEdge(LabelBase):
    """
    Label edge contains the weights that may exist between 2 vertices
    """

    def __init__(self, weight):
        self.weight = weight

    cpdef tuple get_attributes(self):
        return (self.weight, )