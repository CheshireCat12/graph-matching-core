from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelNodeAIDS(LabelBase):
    cdef:
        str symbol
        readonly int symbol_int
        int chem
        int charge
        float x
        float y

    cpdef tuple get_attributes(self)
