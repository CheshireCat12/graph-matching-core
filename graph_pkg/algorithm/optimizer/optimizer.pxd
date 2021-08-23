cdef class Optimizer:

    cdef:
        int size, optimization_turn
        double range_down, range_up
        readonly double[:, ::1] opt_values
        public double[::1] accuracies

    cpdef void update_values(self)
