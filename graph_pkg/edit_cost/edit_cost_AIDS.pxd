from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node
from graph_pkg.edit_cost.metrics cimport dirac_AIDS

ctypedef double (*metricptr)(int, int)

cdef class EditCostAIDS(EditCost):

    cdef:
        int symbol_source, symbol_target
        int valence_source, valence_target

        list metrics_available
        metricptr metric

    cdef double c_cost_insert_node(self, Node node)