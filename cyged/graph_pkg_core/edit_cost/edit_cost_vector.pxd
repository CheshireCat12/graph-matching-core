from cyged.graph_pkg_core.edit_cost.edit_cost cimport EditCost
from cyged.graph_pkg_core.graph.edge cimport Edge
from cyged.graph_pkg_core.graph.node cimport Node
cimport numpy as np

ctypedef double (*metricptr)(double[::1], double[::1])


cdef class EditCostVector(EditCost):

    cdef:
        np.ndarray vec_source, vec_target

        list metrics_available
        # public metricptr metric

    cdef double metric(self, double[::1], double[::1])

    cdef double c_cost_insert_node(self, Node node)

    cdef double c_cost_delete_node(self, Node node)

    cdef double c_cost_substitute_node(self, Node node_src, Node node_trgt)

    cdef double c_cost_insert_edge(self, Edge edge)

    cdef double c_cost_delete_edge(self, Edge edge)

    cdef double c_cost_substitute_edge(self, Edge edge_src, Edge edge_trgt)
