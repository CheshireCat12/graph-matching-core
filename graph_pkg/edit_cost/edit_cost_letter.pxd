from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node

cdef class EditCostLetter(EditCost):

    cdef:
        double x1, y1, x2, y2, result
        int valence

    cdef double _compute_cost_insert_node(self, double x1, double y1) except? -1

    cdef double _compute_cost_substitute_node(self, double x1, double y1, double x2, double y2) except? -1
