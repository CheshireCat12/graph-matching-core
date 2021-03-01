from hierarchical_graph.centrality_measure.centrality_measure cimport CentralityMeasure
from graph_pkg.graph.graph cimport Graph


cdef class PageRank(CentralityMeasure):

    cdef:
        int max_iter
        double damp_factor, tolerance

    cpdef double[::1] calc_centrality_score(self, Graph graph)