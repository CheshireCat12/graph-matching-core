from cyged.graph_pkg_core.edit_cost.edit_cost cimport EditCost
from cyged.graph_pkg_core.graph.graph cimport Graph

cdef class GED:

    cdef:
        int _n, _m
        readonly int[::1] phi
        readonly double[:, ::1] C, C_star

        Graph graph_source, graph_target
        EditCost edit_cost

    cpdef double compute_edit_distance(self,
                                       Graph graph_source,
                                       Graph graph_target,
                                       bint heuristic= *)

    cdef void _init_graphs(self, Graph graph_source, Graph graph_target, bint heuristic)

    cdef void _create_c_matrix(self)

    cdef void _create_c_star_matrix(self)

    cdef double _compute_cost_node_edit(self, int[::1] phi)

    cdef double _compute_cost_edge_edit(self, int[::1] phi)
