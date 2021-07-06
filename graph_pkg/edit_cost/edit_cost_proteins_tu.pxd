from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node
from graph_pkg.edit_cost.metrics cimport dirac

ctypedef double (*metricptr)(int, int)

cdef class EditCostProteinsTU(EditCost):

    cdef:
        int chem_source, chem_target
        int valence_source, valence_target

        list metrics_available
        metricptr metric

    cdef double c_cost_insert_node(self, Node node)

    cdef double c_cost_delete_node(self, Node node)

    cdef double c_cost_substitute_node(self, Node node_src, Node node_trgt)

    cdef double c_cost_insert_edge(self, Edge edge)

    cdef double c_cost_delete_edge(self, Edge edge)

    cdef double c_cost_substitute_edge(self, Edge edge_src, Edge edge_trgt)
