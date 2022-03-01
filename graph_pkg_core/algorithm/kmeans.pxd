import numpy as np
cimport numpy as n

from graph_pkg_core.algorithm.graph_edit_distance cimport GED
from graph_pkg_core.algorithm.matrix_distances cimport MatrixDistances

cdef class Kmeans:

    cdef:
        int n_clusters, max_iter, seed, n_cores
        public double error
        list graphs
        public list centroids
        int[::1] labels, idx_centroids
        MatrixDistances mat_dist

    cpdef tuple init_centroids(self, list graphs)

    cpdef double[:, ::1] compute_distances(self,
                                           list graphs,
                                           list centroids)

    cpdef int[::1] find_closest_cluster(self, double[:, ::1] distances, int[::1] idx_centroids)

    cpdef tuple update_centroids(self,
                                list graphs,
                                list centroids,
                                int[::1] idx_centroids,
                                int[::1] labels)

    cpdef bint are_centroids_equal(self,
                                   list c_centroids,
                                   list old_centroids)

    cpdef double compute_SOD(self,
                             list graphs,
                             list centroids,
                             int[::1] labels)

    cpdef void fit(self, list graphs)
