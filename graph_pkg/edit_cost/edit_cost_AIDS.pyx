cdef class EditCostAIDS(EditCost):

    def __cinit__(self,
                  double c_insert_node,
                  double c_delete_node,
                  double c_insert_edge,
                  double c_delete_edge,
                  str metric_name):
        super().__init__(c_insert_node, c_delete_node, c_insert_edge, c_delete_edge, metric_name)
        self.metrics_available = ['dirac']
        self._init_metric()

    cdef int _init_metric(self) except? -1:
        assert self.metric_name in self.metrics_available, f'The metric {self.metric_name} is not available'

        if self.metric_name == 'dirac':
            self.metric = dirac_AIDS

    cpdef double cost_insert_node(self, Node node) except? -1:
        return self.c_insert_node

    cpdef double cost_delete_node(self, Node node) except? -1:
        return self.c_delete_node

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1:
        self.symbol_source = node1.label.symbol_int
        self.symbol_target = node2.label.symbol_int

        return 0. if self.metric(self.symbol_source, self.symbol_target) == 0. else (self.c_insert_node + self.c_delete_node)

    cpdef double cost_insert_edge(self, Edge edge) except? -1:
        return self.c_insert_edge

    cpdef double cost_delete_edge(self, Edge edge) except? -1:
        return self.c_delete_edge

    cpdef double cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1:
        return 0.
