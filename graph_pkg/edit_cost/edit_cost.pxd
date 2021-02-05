from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node

cdef class EditCost:

    cpdef float cost_insert_node(self, Node node) except? -1

    cpdef float cost_delete_node(self, Node node) except? -1

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1

    cpdef float cost_insert_edge(self, Edge edge) except? -1

    cpdef float cost_delete_edge(self, Edge edge) except? -1

    cpdef float cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1