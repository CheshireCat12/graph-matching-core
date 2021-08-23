cdef class Optimizer:

    def __init__(self, double range_down, range_up, int size, int optimization_turn):
        self.range_down = range_down
        self.range_up = range_up
        self.size = size
        self.optimization_turn = optimization_turn

    cpdef void update_values(self):
        raise NotImplementedError('Function update_values() not implemented!')
