cdef class EditCostLetter(EditCost):

    def __cinit__(self,
                  double c_insert_node,
                  double c_delete_node,
                  double c_insert_edge,
                  double c_delete_edge,
                  str metric_name):
        super().__init__(c_insert_node, c_delete_node, c_insert_edge, c_delete_edge, metric_name)
        self.metrics_available = ['manhattan', 'euclidean']
        self._init_metric()

    cdef int _init_metric(self) except? -1:
        assert self.metric_name in self.metrics_available, f'The metric {self.metric_name} is not available'

        if self.metric_name == 'manhattan':
            self.metric = manhattan_letter
        elif self.metric_name == 'euclidean':
            self.metric = euclidean_letter

    cpdef double cost_insert_node(self, Node node) except? -1:
        return self.c_cost_insert_node(node)

    cdef double c_cost_insert_node(self, Node node):
        return self.c_insert_node

    cpdef double cost_delete_node(self, Node node) except? -1:
        return self.c_cost_delete_node(node)

    cdef double c_cost_delete_node(self, Node node):
        return self.c_delete_node

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1:
        return self.c_cost_substitute_node(node1, node2)

    cdef double c_cost_substitute_node(self, Node node_src, Node node_trgt):
        return self.metric(node_src.label.x, node_src.label.y, node_trgt.label.x, node_trgt.label.y)

    cpdef double cost_insert_edge(self, Edge edge) except? -1:
        return self.c_cost_insert_edge(edge)

    cdef double c_cost_insert_edge(self, Edge edge):
        return self.c_insert_edge

    cpdef double cost_delete_edge(self, Edge edge) except? -1:
        return self.c_cost_delete_edge(edge)

    cdef double c_cost_delete_edge(self, Edge edge):
        return self.c_delete_edge

    cpdef double cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1:
        return self.c_cost_substitute_edge(edge1, edge2)

    cdef double c_cost_substitute_edge(self, Edge edge_src, Edge edge_trgt):
        return 0.
