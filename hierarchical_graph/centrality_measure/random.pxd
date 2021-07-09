from hierarchical_graph.centrality_measure.centrality_measure cimport CentralityMeasure
from graph_pkg.graph.graph cimport Graph


cdef class Random(CentralityMeasure):

    cpdef double[::1] calc_centrality_score(self, Graph graph)
