from hierarchical_graph.centrality_measure.centrality_measure cimport CentralityMeasure
from graph_pkg.graph.graph cimport Graph
from hierarchical_graph.utils.sigma_js cimport SigmaJS

cdef class HierarchicalGraph:

    cdef:
        list level_graphs
        CentralityMeasure measure
        SigmaJS sigma_js

    cpdef void create_hierarchy(self, strategy=*)