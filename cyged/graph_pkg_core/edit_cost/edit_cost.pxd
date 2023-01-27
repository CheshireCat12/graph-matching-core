from cyged.graph_pkg_core.graph.edge cimport Edge
from cyged.graph_pkg_core.graph.node cimport Node


cdef class EditCost:

    cdef:
        readonly double c_insert_node
        readonly double c_delete_node
        readonly double c_insert_edge
        readonly double c_delete_edge
        readonly str metric_name
        double alpha_node, alpha_edge
        # change_alpha is used during the serialization of the EditCost to
        # check which alpha value is used
        bint change_alpha

    cdef void _init_alpha(self, double alpha)

    cpdef void update_alpha(self, double alpha)

    cdef int _init_metric(self) except? -1

    cpdef double cost_insert_node(self, Node node) except? -1

    cpdef double cost_delete_node(self, Node node) except? -1

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1

    cpdef double cost_insert_edge(self, Edge edge) except? -1

    cpdef double cost_delete_edge(self, Edge edge) except? -1

    cpdef double cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1

    cdef double c_cost_insert_node(self, Node node)

    cdef double c_cost_delete_node(self, Node node)

    cdef double c_cost_substitute_node(self, Node node_src, Node node_trgt)

    cdef double c_cost_insert_edge(self, Edge edge)

    cdef double c_cost_delete_edge(self, Edge edge)

    cdef double c_cost_substitute_edge(self, Edge edge_src, Edge edge_trgt)
