from graph_pkg_core.algorithm.optimizer.optimizer cimport Optimizer

cdef class GeneticAlgorithm(Optimizer):

    cdef:
        int round_val
        double p_crossover, p_mutation, gamma

    cpdef void update_values(self)
