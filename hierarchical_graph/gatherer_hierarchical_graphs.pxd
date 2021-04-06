from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs
from graph_pkg.utils.constants cimport MEASURES

cdef class GathererHierarchicalGraphs:

    cdef:
        list graphs_train, graphs_val, graphs_test
        readonly list labels_train, labels_val, labels_test
        readonly HierarchicalGraphs h_graphs_train, h_graphs_val, h_graphs_test
        readonly CoordinatorClassifier coordinator