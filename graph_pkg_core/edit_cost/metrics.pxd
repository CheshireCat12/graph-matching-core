
cdef double manhattan_letter(double x1, double y1, double x2, double y2)

cdef double euclidean_letter(double x1, double y1, double x2, double y2)

cdef double dirac_AIDS(int symbol_source, int symbol_target)

cdef double dirac_mutagenicity(int chem_source, int chem_target)

cdef double dirac_NCI1(int chem_source, int chem_target)

cdef double dirac(int source, int target)

cdef double euclidean_vector(double[::1] vec_src, double[::1] vec_trgt)
