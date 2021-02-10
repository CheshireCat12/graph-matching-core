cdef class EditCost:

    def __cinit__(self,
                  double c_insert_node,
                  double c_delete_node,
                  double c_insert_edge,
                  double c_delete_edge,
                  str metric_name):
        self.c_insert_node = c_insert_node
        self.c_delete_node = c_delete_node
        self.c_insert_edge = c_insert_edge
        self.c_delete_edge = c_delete_edge
        self._metric_name = metric_name

    cdef int _init_metric(self) except? -1:
        raise NotImplementedError

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
