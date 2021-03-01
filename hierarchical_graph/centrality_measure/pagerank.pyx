import numpy as np
cimport numpy as np
import scipy.sparse as sprs


cdef class PageRank(CentralityMeasure):

    def __init__(self, double damp_factor=0.85, int max_iter=100, double tolerance=1e-6):
        super(PageRank, self).__init__()

        self.damp_factor = 0.85
        self.max_iter=100
        self.tolerance = tolerance

    cpdef double[::1] calc_centrality_score(self, Graph graph):
        """
        Compute the page rank from the given adjacency matrix.
        Code from : https://asajadi.github.io/fast-pagerank/
        
        :param graph: 
        :return: np.array[double] - PageRank score of the nodes
        """
        cdef:
            int n, iteration

        n, *_ = graph.adjacency_matrix.shape
        adj = np.asarray(graph.adjacency_matrix, dtype=np.int32)

        r = adj.sum(axis=1)

        k = r.nonzero()[0]
        D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

        personalize = np.ones(n)
        personalize = personalize.reshape(n, 1)
        s = (personalize / personalize.sum()) * n

        z_T = (((1 - self.damp_factor) * (r != 0) + (r == 0)) / n)[np.newaxis, :]
        W = self.damp_factor * adj.T @ D_1

        x = s
        old_x = np.zeros((n, 1))

        iteration = 0
        while np.linalg.norm(x - old_x) > self.tolerance:
            old_x = x
            x = W @ x + s @ (z_T @ x)

            iteration += 1
            if iteration >= self.max_iter:
                break

        x = x / x.sum()

        return x.reshape(-1)


