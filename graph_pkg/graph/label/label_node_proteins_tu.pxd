from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelNodeProteinsTU(LabelBase):
    cdef:
        readonly int chem

    cpdef tuple get_attributes(self)
