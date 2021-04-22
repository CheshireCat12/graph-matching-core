from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs
from graph_pkg.utils.constants cimport MEASURES
from hierarchical_graph.centrality_measure.centrality_measure cimport CentralityMeasure

cdef class GathererHierarchicalGraphs:

    cdef:
        CentralityMeasure measure
        readonly list graphs_train, graphs_val, graphs_test, percentages, aggregation_graphs
        readonly list labels_train, labels_val, labels_test, aggregation_labels
        readonly HierarchicalGraphs h_graphs_train, h_graphs_val, h_graphs_test, h_aggregation_graphs
        readonly CoordinatorClassifier coordinator

    cpdef list k_fold_validation(self, int cv=*)