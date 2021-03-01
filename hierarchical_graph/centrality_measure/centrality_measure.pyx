cdef class CentralityMeasure:

    def __init__(self, name):
        self.name = name

    cpdef double[::1] calc_centrality_score(self, Graph graph):
        raise NotImplementedError('Centrality score not implemented!')
