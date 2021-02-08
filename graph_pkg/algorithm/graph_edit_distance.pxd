from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.graph cimport Graph
from graph_pkg.graph.node cimport Node

cdef class GED:

    cdef:
        readonly double[:, ::1] C, C_star

        Graph graph_source, graph_target
        EditCost edit_cost

    cdef void _create_c_matrix(self)

    cdef void _create_c_star_matrix(self)

    cpdef double compute_distance_between_graph(self, Graph graph_source, Graph graph_target)