from graph_pkg.graph.label.label_node cimport LabelNode

cdef class LabelNodeAIDS(LabelNode):
    cdef:
        str symbol
        int chem
        int charge
        float x
        float y

    cpdef tuple get_attributes(self)