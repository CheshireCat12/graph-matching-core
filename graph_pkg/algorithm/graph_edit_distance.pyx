cdef class GED:

    def __cinit__(self, EditCost edit_cost):
        self.edit_cost = edit_cost

    cpdef float compute_distance_between_graph(self, Graph graph1, Graph graph2):
        cdef:
            int m, n
            int[:, :] adj_mat1, adj_mat2

        m = len(graph1)
        n = len(graph2)

        adj_mat1 = graph1.adjacency_matrix
        adj_mat2 = graph2.adjacency_matrix
