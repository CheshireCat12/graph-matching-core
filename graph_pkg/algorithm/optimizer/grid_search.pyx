from itertools import product

import numpy as np
cimport numpy as np

cdef class GridSearch(Optimizer):

    def __init__(self, double range_down, range_up, int size, int optimization_turn=1):
        super().__init__(range_down, range_up, size, optimization_turn)

        coeff_range = [i/10 for i in range(0, 10)]

        self.opt_values = np.array(list(product(coeff_range, repeat=size)))[1:]
        self.accuracies = np.zeros(len(self.opt_values))

    cpdef void update_values(self):
        pass
