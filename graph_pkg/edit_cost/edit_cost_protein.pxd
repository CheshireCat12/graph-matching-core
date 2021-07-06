from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node
from graph_pkg.algorithm.levenshtein_distance cimport LevenshteinDistance


cdef class EditCostProtein(EditCost):

    cdef:
        int type_source, type_target
        str sequence_source, sequence_target
        double string_edit_substitute, string_edit_insert, string_edit_delete

        list metrics_available
        LevenshteinDistance metric

    cdef double c_cost_insert_node(self, Node node)

    cdef double c_cost_delete_node(self, Node node)

    cdef double c_cost_substitute_node(self, Node node_src, Node node_trgt)

    cdef double c_cost_insert_edge(self, Edge edge)

    cdef double c_cost_delete_edge(self, Edge edge)

    cdef double c_cost_substitute_edge(self, Edge edge_src, Edge edge_trgt)
