from hierarchical_graph.centrality_measure.centrality_measure cimport CentralityMeasure
from graph_pkg.graph.graph cimport Graph
from hierarchical_graph.utils.sigma_js cimport SigmaJS

cdef class HierarchicalGraph:

    cdef:
        list level_graphs
        CentralityMeasure measure
        SigmaJS sigma_js

    cpdef list create_hierarchy_percent(self, list graphs,
                                        double percentage_remaining=*,
                                        str deletion_strategy=*,
                                        bint verbose=*)

    cpdef void _update_graph_compute_once(self, Graph graph,
                                          int num_nodes_to_del)

    cpdef void _update_graph_recomputing(self, Graph graph,
                                            int num_nodes_to_del)

    # cpdef void _save_graph_to_js(self, Graph graph,
    #                              int num_nodes_to_del,
    #                              double[::1] centrality_score)
