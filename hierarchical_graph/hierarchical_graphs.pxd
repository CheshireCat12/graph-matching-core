from hierarchical_graph.centrality_measure.centrality_measure cimport CentralityMeasure
from graph_pkg.graph.graph cimport Graph


cdef class HierarchicalGraphs:

    cdef:
        readonly dict hierarchy
        list original_graphs
        CentralityMeasure measure
        str deletion_strategy
        bint verbose

    cpdef void _create_hierarchy_of_graphs(self)

    cpdef list _reduce_graphs(self, list graphs, double percentage_remaining=*)

    cpdef void _update_graph_compute_once(self, Graph graph,
                                          int num_nodes_to_del)

    cpdef void _update_graph_recomputing(self, Graph graph,
                                            int num_nodes_to_del)
