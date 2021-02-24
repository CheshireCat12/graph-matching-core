import numpy as np
cimport numpy as np
import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg



cpdef double[::1] pagerank_power(int[:, ::1] adjacency_mat, double damp_fact=0.85, int max_iter=100, double tolerance=1e-6):
    """
    Compute the page rank from the given adjacency matrix.
    Code from : https://asajadi.github.io/fast-pagerank/
    
    :param adjacency_mat: 
    :param damp_fact: 
    :param tolerance: 
    :return: np.array[double] - PageRank score of the nodes
    """
    cdef:
        int n, iteration

    n, *_ = adjacency_mat.shape
    adj = np.asarray(adjacency_mat, dtype=np.int32)

    r = adj.sum(axis=1)

    k = r.nonzero()[0]
    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    personalize = np.ones(n)
    personalize = personalize.reshape(n, 1)
    s = (personalize / personalize.sum()) * n

    z_T = (((1 - damp_fact) * (r != 0) + (r == 0)) / n)[np.newaxis, :]
    W = damp_fact * adj.T @ D_1

    x = s
    old_x = np.zeros((n, 1))

    iteration = 0
    while np.linalg.norm(x - old_x) > tolerance:
        old_x = x
        x = W @ x + s @ (z_T @ x)

        iteration += 1
        if iteration >= max_iter:
            break

    x = x / x.sum()

    return x.reshape(-1)


