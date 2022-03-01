from graph_pkg_core.algorithm.optimizer.optimizer cimport Optimizer

cdef class GridSearch(Optimizer):

    cpdef void update_values(self)
