cdef class EditCostProtein(EditCost):
    def __init__(self,
                 double c_insert_node,
                 double c_delete_node,
                 double c_insert_edge,
                 double c_delete_edge,
                 str metric_name,
                 double string_edit_substitute,
                 double string_edit_insert,
                 double string_edit_delete,
                 double alpha=-1.
                 ):
        super().__init__(c_insert_node, c_delete_node, c_insert_edge, c_delete_edge, metric_name, alpha)
        self.metrics_available = ['sed']
        self.string_edit_substitute = string_edit_substitute
        self.string_edit_insert = string_edit_insert
        self.string_edit_delete = string_edit_delete

        self._init_metric()

    cdef int _init_metric(self) except? -1:
        assert self.metric_name in self.metrics_available, f'The metric {self.metric_name} is not available'

        if self.metric_name == 'sed':
            sed = LevenshteinDistance()
            self.metric = LevenshteinDistance()  # sed.compute.... #  sed_protein

    cpdef double cost_insert_node(self, Node node) except? -1:
        return self.c_cost_insert_node(node)

    cdef double c_cost_insert_node(self, Node node):
        return self.alpha_node * self.c_insert_node  # * len(node.label.sequence)

    cpdef double cost_delete_node(self, Node node) except? -1:
        return self.c_cost_delete_node(node)

    cdef double c_cost_delete_node(self, Node node):
        return self.alpha_node * self.c_delete_node

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1:
        """
        Compute the substitution of the two given nodes.
        It checks if the chemical symbols are the same.
        If they are it returns 0.
        Otherwise it returns 2*Tau_node

        See Kaspar's thesis (p.88 - AIDS and Mutagenicity Graphs)
        :param node1: 
        :param node2: 
        :return: double - Cost to substitute node
        """
        return self.c_cost_substitute_node(node1, node2)

    cdef double c_cost_substitute_node(self, Node node_src, Node node_trgt):
        self.type_source = node_src.label.type_
        self.type_target = node_trgt.label.type_
        self.sequence_source = node_src.label.sequence
        self.sequence_target = node_trgt.label.sequence

        if self.type_source == self.type_target:
            cost = self.metric.compute_string_edit_distance(self.sequence_source, self.sequence_target,
                                                            self.string_edit_substitute,
                                                            self.string_edit_insert, self.string_edit_delete)
        else:
            cost = self.c_insert_node + self.c_delete_node

        return self.alpha_node * cost

    cpdef double cost_insert_edge(self, Edge edge) except? -1:
        return self.c_cost_insert_edge(edge)

    cdef double c_cost_insert_edge(self, Edge edge):
        return self.alpha_edge * self.c_insert_edge

    cpdef double cost_delete_edge(self, Edge edge) except? -1:
        return self.c_cost_delete_edge(edge)

    cdef double c_cost_delete_edge(self, Edge edge):
        return self.alpha_edge * self.c_delete_edge

    cpdef double cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1:
        return self.c_cost_substitute_edge(edge1, edge2)

    cdef double c_cost_substitute_edge(self, Edge edge_src, Edge edge_trgt):
        return self.alpha_edge * 0.

    def __reduce__(self):
        d = dict()
        d['c_insert_node'] = self.c_insert_node
        d['c_delete_node'] = self.c_delete_node
        d['c_insert_edge'] = self.c_insert_edge
        d['c_delete_edge'] = self.c_delete_edge
        d['metric_name'] = self.metric_name
        d['string_edit_substitute'] = self.string_edit_substitute
        d['string_edit_insert'] = self.string_edit_insert
        d['string_edit_delete'] = self.string_edit_delete
        d['alpha'] = self.alpha_node if self.change_alpha else -1

        return (rebuild, (d,))

def rebuild(data):
    cdef EditCost edit_cost
    edit_cost = EditCostProtein(**data)

    return edit_cost
