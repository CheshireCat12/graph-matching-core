from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.graph cimport Graph

cdef class GED:

    cdef:
        Graph graph1, graph2
        EditCost edit_cost

    cpdef float compute_distance_between_graph(self, Graph graph1, Graph graph2)