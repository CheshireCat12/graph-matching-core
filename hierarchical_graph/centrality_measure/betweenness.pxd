from hierarchical_graph.centrality_measure.centrality_measure cimport CentralityMeasure
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.graph cimport Graph
from graph_pkg.graph.node cimport Node


cdef class Betweenness(CentralityMeasure):

    cpdef double[::1] calc_centrality_score(self, Graph graph)