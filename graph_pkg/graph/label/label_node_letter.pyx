from graph_pkg.graph.label.label_node import LabelNode

cdef class LabelNodeLetter(LabelNode):

    def __cinit__(self, x, y):
        self.x = x
        self.y = y

    cpdef tuple get_attributes(self):
        return self.x, self.y
