import numpy as np
cimport numpy as np


cdef class EditCostGNNEmbedding(EditCost):

    def __init__(self,
                 double c_insert_node,
                 double c_delete_node,
                 double c_insert_edge,
                 double c_delete_edge,
                 str metric_name,
                 double alpha=-1.):
        super().__init__(c_insert_node, c_delete_node, c_insert_edge, c_delete_edge, metric_name, alpha)
        self.metrics_available = ['euclidean']
        self._init_metric()

    cdef int _init_metric(self) except? -1:
        assert self.metric_name in self.metrics_available, f'The metric {self.metric_name} is not available'

        if self.metric_name == 'euclidean':
            self.metric = euclidean_vector

    cpdef double cost_insert_node(self, Node node) except? -1:
        return self.c_cost_insert_node(node)

    cdef double c_cost_insert_node(self, Node node):
        return self.alpha_node * self.c_insert_node

    cpdef double cost_delete_node(self, Node node) except? -1:
        return self.c_cost_delete_node(node)

    cdef double c_cost_delete_node(self, Node node):
        return self.alpha_node * self.c_delete_node

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1:
        """
        Compute the substitution of the two given nodes.
        It computes the euclidean distance between the two input vectors
        We ensure the cost being in [0, 2tau] with the sigmoid function

        :param node1: 
        :param node2: 
        :return: double - Cost to substitute node
        """
        return self.c_cost_substitute_node(node1, node2)

    cdef double c_cost_substitute_node(self, Node node_src, Node node_trgt):
        cdef:
            double dist, sub_cost, alpha, sigma
        self.vec_source = node_src.label.vector
        self.vec_target = node_trgt.label.vector

        dist = self.metric(self.vec_source, self.vec_target)
        # sigmoid_term = 1 / (2*self.c_insert_node)
        # alpha = 1
        # sigma = 5 #np.log(1/2)
        #
        # sub_cost = 1 / (sigmoid_term + np.exp(-alpha * dist + sigma))
        #
        # return self.alpha_node * sub_cost
        return self.alpha_node * np.min([dist, 2*self.c_insert_node])

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
        d['alpha'] = self.alpha_node if self.change_alpha else -1

        return (rebuild, (d,))

def rebuild(data):
    cdef EditCost edit_cost
    edit_cost = EditCostGNNEmbedding(**data)

    return edit_cost
