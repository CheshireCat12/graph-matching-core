import numpy as np
cimport numpy as np


cdef class Kmeans:

    def __init__(self, GED ged, list graphs, int max_iter=100, int seed=42, n_cores=-1):
        self.graphs = graphs
        self.n_clusters = 1
        self.max_iter = max_iter
        self.seed = seed

        self.mat_dist = MatrixDistances(ged, parallel=True, verbose=False)
        self.n_cores = n_cores

        self._init_distances(self.graphs)

    cpdef void _init_distances(self, list graphs):
        self.full_distances = self.compute_distances(graphs,
                                                np.arange(len(graphs), dtype=np.int32))
        np.savetxt(f'./tmp_mat_distance/{self.seed}_{len(graphs)}.csv',
                   np.array(self.full_distances),
                   delimiter=',')

    cpdef void set_n_cluster_and_seed(self,int n_clusters, int new_seed):
        self.n_clusters = n_clusters
        self.seed = new_seed

    cpdef int[::1] init_centroids(self, list graphs):
        cdef:
            list centroids

        np.random.seed(self.seed)
        idx_centroids = np.random.permutation(len(graphs))[:self.n_clusters]

        return np.array(idx_centroids, dtype=np.int32)

    cpdef double[:, ::1] compute_distances(self,
                                           list graphs,
                                           int[::1] idx_centroids):
        cdef:
            list centroids = [graphs[idx] for idx in idx_centroids]

        return self.mat_dist.calc_matrix_distances(graphs,
                                                   centroids,
                                                   heuristic=True,
                                                   num_cores=self.n_cores)

    cpdef int[::1] find_closest_cluster(self,
                                        double[:, :] distances,
                                        int[::1] idx_centroids):
        closest_pts = np.array(np.argmin(distances, axis=1), dtype=np.int32)

        # Change the cluster of the centroid by hand to be sure that at least one point is
        # in each cluster
        for idx, idx_c in enumerate(idx_centroids):
            closest_pts[idx_c] = idx

        return closest_pts

    cpdef int[::1] update_centroids(self,
                                double[:, :] distances,
                                int[::1] idx_centroids,
                                int[::1] labels):
        cdef:
            list new_idx_centroids = []
            double[:, :] intra_cls_distances

        for k in range(self.n_clusters):
            cls_indices = np.where(np.array(labels)==k)[0]
            # print(cls_indices)
            # print('%%%%')
            # print(k)
            # graphs_per_cls = [graphs[idx] for idx in cls_indices]

            # print(graphs_per_cls)

            # intra_cls_distances = self.compute_distances(graphs_per_cls,
            #                                              np.arange(len(graphs_per_cls), dtype=np.int32))
            intra_cls_distances = np.asarray(distances)[cls_indices,:][:,cls_indices]

            # print(np.sum(distances, axis=0))
            try:
                idx_new_centroid = np.argmin(np.sum(intra_cls_distances, axis=1))
            except ValueError:
                print(np.array(labels))
                print(cls_indices)


            # print('distances')
            # print(np.sum(intra_cls_distances, axis=1))
            # print(np.array(intra_cls_distances[0]))
            # print('num per cluster', len(np.array(intra_cls_distances[0])))

            new_idx_centroids.append(cls_indices[idx_new_centroid])
            # print('##############################')
        return np.array(new_idx_centroids, dtype=np.int32)

    cpdef bint are_centroids_equal(self, int[::1] cur_centroids, int[::1] old_centroids):
        return np.array_equal(cur_centroids, old_centroids)

    cpdef double compute_SOD(self, double[:, :] distances, int[::1] idx_centroids, int[::1] labels):
        cdef:
            double error = 0
            double[:, :] intra_cls_distances

        for k in range(self.n_clusters):
            cls_indices = np.where(np.array(labels) == k)[0]

            # graphs_per_cls = [graphs[idx] for idx in cls_indices]
            # intra_cls_distances = self.compute_distances(graphs_per_cls,
            #                                              np.arange(len(graphs_per_cls), dtype=np.int32))

            intra_cls_distances = np.asarray(distances)[cls_indices,:][:,cls_indices]

            error += np.sum(intra_cls_distances) # * len(graphs_per_cls)
        return error # / self.n_clusters

    cpdef void fit(self):
        cdef:
            int[::1] old_centroids
            double[:, :] full_distances, distances

        # self.idx_centroids, self.centroids = self.init_centroids(graphs)
        self.idx_centroids = self.init_centroids(self.graphs)
        # print(f'----> {self.seed}')


        self.error = float('inf')
        self.labels = np.array([], dtype=np.int32)

        for turn in range(self.max_iter):

            # distances = self.compute_distances(graphs,
            #                                    self.idx_centroids)
            distances = np.asarray(self.full_distances)[:,self.idx_centroids]
            # print(np.array(distances))
            # print(np.array(distances)[33])
            # print(np.argmin(np.array(distances), axis=1))

            self.labels = self.find_closest_cluster(distances, self.idx_centroids)

            error_tmp = self.compute_SOD(self.full_distances, self.idx_centroids, self.labels)

            # self.error = error_tmp
            if error_tmp <= self.error:
                self.error = error_tmp
            else:
                self.idx_centroids = old_idx_centroids
                self.labels = old_labels
                # print('break')
                break
            old_idx_centroids = self.idx_centroids
            old_labels = self.labels
            # print('temp error', self.error)
            # print(np.array(self.idx_centroids))
            self.idx_centroids = self.update_centroids(self.full_distances,
                                                       self.idx_centroids,
                                                       self.labels)

            if self.are_centroids_equal(self.idx_centroids, old_idx_centroids):
                break
        # print(np.array(self.idx_centroids))
        self.error = self.compute_SOD(self.full_distances, self.idx_centroids, self.labels)
        # print(f'final error: {self.error}')
        # print([self.graphs[idx].name for idx in self.idx_centroids])