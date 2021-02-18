cdef class EditCost:

    def __init__(self,
                  double c_insert_node,
                  double c_delete_node,
                  double c_insert_edge,
                  double c_delete_edge,
                  str metric_name,
                  double alpha=-1.):
        self.c_insert_node = c_insert_node
        self.c_delete_node = c_delete_node
        self.c_insert_edge = c_insert_edge
        self.c_delete_edge = c_delete_edge
        self.metric_name = metric_name
        self._init_alpha(alpha)

    cdef void _init_alpha(self, double alpha):
        assert alpha == -1. or 0. <= alpha <= 1., f'The parameter alpha is not valid!\nIt must be 0 <= alpha <= 1'
        if 0. <= alpha <= 1.:
            self.alpha_node = alpha
            self.alpha_edge = 1. - alpha
        else:
            self.alpha_node = 1.
            self.alpha_edge = 1.

    def __repr__(self):
        return f'cost_node{self.c_insert_node}_cost_edge{self.c_insert_edge}_alpha{self.alpha_node}'

    def __str__(self):
        return f'Cost Insert/Delete Node: {self.c_insert_node};\n' \
               f'Cost Insert/Delete Edge: {self.c_insert_edge};\n' \
               f'Alpha Node: {self.alpha_node}, Alpha Edge: {self.alpha_edge};\n' \
               f'Metric Function: {self.metric_name}'

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

    cdef double c_cost_insert_node(self, Node node):
        pass

    cdef double c_cost_delete_node(self, Node node):
        pass

    cdef double c_cost_substitute_node(self, Node node_src, Node node_trgt):
        pass

    cdef double c_cost_insert_edge(self, Edge edge):
        pass

    cdef double c_cost_delete_edge(self, Edge edge):
        pass

    cdef double c_cost_substitute_edge(self, Edge edge_src, Edge edge_trgt):
        pass
