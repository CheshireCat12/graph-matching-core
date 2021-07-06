from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelNodeProtein(LabelBase):
    cdef:
        readonly int type_
        int aa_length
        readonly str sequence

    cpdef tuple get_attributes(self)
