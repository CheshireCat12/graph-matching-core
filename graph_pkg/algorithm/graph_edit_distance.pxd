from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.graph cimport Graph
from graph_pkg.graph.node cimport Node
from graph_pkg.graph.edge cimport Edge

cdef class GED:

    cdef:
        int _n, _m
        double alpha_node, alpha_edge
        readonly int[::1] phi
        readonly double[:, ::1] C, C_star

        Graph graph_source, graph_target
        EditCost edit_cost

    cpdef double compute_edit_distance(self,
                                       Graph graph_source,
                                       Graph graph_target,
                                       double alpha= *,
                                       bint heuristic= *)

    cdef void _init_graphs(self, Graph graph_source, Graph graph_target, bint heuristic)

    cdef void _init_alpha(self, double alpha)

    cdef void _create_c_matrix(self)

    cdef void _create_c_star_matrix(self)

    cdef double _compute_cost_node_edit(self, int[::1] phi)

    cdef double _compute_cost_edge_edit(self, int[::1] phi)
