cdef class EditCostMutagenicity(EditCost):

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
        assert self._metric_name in self.metrics_available, f'The metric {self._metric_name} is not available'

        if self._metric_name == 'dirac':
            self.metric = dirac_mutagenicity

    cpdef double cost_insert_node(self, Node node) except? -1:
        return self._c_insert_node

    cpdef double cost_delete_node(self, Node node) except? -1:
        return self._c_delete_node

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1:
        self.chem_source = node1.label.chem_int
        self.chem_target = node2.label.chem_int

        return self.metric(self.chem_source, self.chem_target)

    cpdef double cost_insert_edge(self, Edge edge) except? -1:
        return self._c_insert_edge

    cpdef double cost_delete_edge(self, Edge edge) except? -1:
        return self._c_delete_edge

    cpdef double cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1:
        return 0.
