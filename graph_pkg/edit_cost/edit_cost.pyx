cdef class EditCost:

    def __cinit__(self):
        pass

    cpdef double cost_insert_node(self, Node node) except? -1:
        raise NotImplementedError

    cpdef double cost_delete_node(self, Node node) except? -1:
        raise NotImplementedError

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1:
        raise NotImplementedError

    cpdef double cost_insert_edge(self, Edge edge) except? -1:
        raise NotImplementedError

    cpdef double cost_delete_edge(self, Edge edge) except? -1:
        raise NotImplementedError

    cpdef double cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1:
        raise NotImplementedError
