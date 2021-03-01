from graph_pkg.graph.graph cimport Graph


cdef class CentralityMeasure:

    cdef readonly str name

    cpdef double[::1] calc_centrality_score(self, Graph graph)

