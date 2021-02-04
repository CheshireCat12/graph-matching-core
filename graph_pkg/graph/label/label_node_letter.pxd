from graph_pkg.graph.label.label_node cimport LabelNode

cdef class LabelNodeLetter(LabelNode):
    cdef:
        float x
        float y

    cpdef tuple get_attributes(self)