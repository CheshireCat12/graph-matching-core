import numpy as np
cimport numpy as np

from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.graph.graph cimport Graph


cdef class MatrixDistances:

    cdef:
        list graphs
        GED ged


    cpdef void create_matrix_distance_diagonal(self)
