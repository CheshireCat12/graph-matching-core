from libc.math cimport pow as c_pow
from libc.math cimport abs as c_abs
from libc.math cimport sqrt as c_sqrt
import numpy as np
cimport numpy as np

cdef double manhattan_letter(double x1, double y1, double x2, double y2):
    return c_abs(x1 - x2) + c_abs(y1 - y2)

cdef double euclidean_letter(double x1, double y1, double x2, double y2):
    return c_sqrt(c_pow(x1 - x2, 2) + c_pow(y1 - y2, 2))

cdef double dirac_AIDS(int symbol_source, int symbol_target):
    return float(symbol_source != symbol_target)

cdef double dirac_mutagenicity(int chem_source, int chem_target):
    return float(chem_source != chem_target)

cdef double dirac_NCI1(int chem_source, int chem_target):
    return float(chem_source != chem_target)

cdef double dirac(int source, int target):
    return float(source != target)

cdef double euclidean_vector(double[::1] vec_src, double[::1] vec_trgt):
    return np.linalg.norm(np.array(vec_src) - np.array(vec_trgt))
