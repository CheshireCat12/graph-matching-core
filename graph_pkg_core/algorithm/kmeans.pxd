import numpy as np
cimport numpy as n

from graph_pkg_core.algorithm.graph_edit_distance cimport GED
from graph_pkg_core.algorithm.matrix_distances cimport MatrixDistances

cdef class Kmeans:

    cdef:
        int n_clusters, max_iter, seed, n_cores
        public double error
        list graphs
        public int[::1] labels, idx_centroids
        double[:, :] full_distances
        MatrixDistances mat_dist

    cpdef void _init_distances(self, list graphs)

    cpdef void set_n_cluster_and_seed(self, int n_clusters, int new_seed)

    cpdef int[::1] init_centroids(self, list graphs)

    cpdef double[:, ::1] compute_distances(self,
                                           list graphs,
                                           int[::1] centroids)

    cpdef int[::1] find_closest_cluster(self,
                                        double[:, :] distances,
                                        int[::1] idx_centroids)

    cpdef int[::1] update_centroids(self,
                                double[:, :] distances,
                                int[::1] idx_centroids,
                                int[::1] labels)

    cpdef bint are_centroids_equal(self,
                                   int[::1] c_centroids,
                                   int[::1] old_centroids)

    cpdef double compute_SOD(self,
                             double[:, :] distances,
                             int[::1] idx_centroids,
                             int[::1] labels)

    cpdef void fit(self)
