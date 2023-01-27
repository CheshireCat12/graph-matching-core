from libc.math cimport fmin
from libc.math cimport pow as c_pow
from libc.math cimport sqrt as c_sqrt

cimport cython

from cyged.graph_pkg_core.edit_cost.edit_cost cimport EditCost
from cyged.graph_pkg_core.graph.edge cimport Edge
from cyged.graph_pkg_core.graph.node cimport Node


cdef class EditCostVector(EditCost):

    def __init__(self,
                 double c_insert_node,
                 double c_delete_node,
                 double c_insert_edge,
                 double c_delete_edge,
                 str metric_name,
                 double alpha=-1.):
        super().__init__(c_insert_node, c_delete_node, c_insert_edge, c_delete_edge, metric_name, alpha)
        self.metrics_available = ['euclidean']

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double metric(self, double[::1] vec_src, double[::1] vec_trgt):

        cdef:
            int N
            double sum_pow = 0.
        N = vec_src.shape[0]
        for idx in range(N):
            sum_pow += c_pow(vec_src[idx] - vec_trgt[idx], 2)

        return c_sqrt(sum_pow)


    cpdef double cost_insert_node(self, Node node) except? -1:
        """
        Compute the cost to insert a node in the graph
        cost = alpha_node * cost insertion
        
        Args:
            node: 

        Returns:

        """
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

        return self.alpha_node * fmin(dist, 2*self.c_insert_node)

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
    edit_cost = EditCostVector(**data)

    return edit_cost
