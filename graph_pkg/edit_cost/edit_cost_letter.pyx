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
        return self.c_insert_node

    cpdef double cost_delete_node(self, Node node) except? -1:
        return self.c_delete_node

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1:
        # self.x1, self.y1 = node1.label.get_attributes()
        # self.x2, self.y2 = node2.label.get_attributes()
        # cdef:
        #     double x1, y1, x2, y2
        #
        # x1 = node1.label.x
        # y1 = node1.label.y
        # x2 = node2.label.x
        # y2 = node2.label.y

        # return self.metric(x1, y1, x2, y2)
        return self.metric(node1.label.x, node1.label.y, node2.label.x, node2.label.y) # self.x1, self.y1, self.x2, self.y2)

    cpdef double cost_insert_edge(self, Edge edge) except? -1:
        return self.c_insert_edge

    cpdef double cost_delete_edge(self, Edge edge) except? -1:
        return self.c_delete_edge

    cpdef double cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1:
        return 0.
