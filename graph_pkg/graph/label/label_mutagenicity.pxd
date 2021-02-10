from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LabelNodeMutagenicity(LabelBase):
    cdef:
        str chem
        readonly int chem_int

    cpdef tuple get_attributes(self)
