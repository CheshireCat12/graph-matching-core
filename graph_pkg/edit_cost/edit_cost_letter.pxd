from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node

cdef class EditCostLetter(EditCost):

    cdef:
        float x1, y1, x2, y2, result
        int valence

    cdef float _compute_cost_insert_node(self, float x1, float y1) except? -1

    cdef float _compute_cost_substitute_node(self, float x1, float y1, float x2, float y2) except? -1
