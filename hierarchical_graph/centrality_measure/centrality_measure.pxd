from graph_pkg.graph.graph cimport Graph


cdef class CentralityMeasure:

    cpdef double[::1] calc_centrality_score(self, Graph graph)

