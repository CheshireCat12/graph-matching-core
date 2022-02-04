import numpy as np

cdef class Kmeans:

    def __init__(self, int n_clusters, GED ged, int max_iter=100, int seed=42, n_cores=-1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

        self.mat_dist = MatrixDistances(ged, parallel=True, verbose=False)
        self.n_cores = n_cores

    cpdef list init_centroids(self, list graphs):
        cdef:
            list centroids

        np.random.seed(self.seed)
        random_idx = np.random.permutation(len(graphs))[:self.n_clusters]
        centroids = [graphs[idx] for idx in random_idx]

        return centroids

    cpdef double[:, ::1] compute_distances(self,
                                           list graphs,
                                           list centroids):
        return self.mat_dist.calc_matrix_distances(graphs,
                                                   centroids,
                                                   heuristic=True,
                                                   num_cores=self.n_cores)

    cpdef int[::1] find_closest_cluster(self, double[:, ::1] distances):
        # breakpoint()
        test = np.array(np.argmin(distances, axis=1), dtype=np.int32)
        print(test)

        return test

    cpdef list update_centroids(self,
                                list graphs,
                                list centroids,
                                int[::1] labels):
        cdef:
            list new_centroids = []
            double[:, ::1] intra_cls_distances

        for k in range(self.n_clusters):
            cls_indices = np.where(np.array(labels)==k)[0]
            # print(class_indices)
            print(k)
            graphs_per_cls = [graphs[idx] for idx in cls_indices]

            print(graphs_per_cls)

            intra_cls_distances = self.compute_distances(graphs_per_cls, graphs_per_cls)

            # print(np.sum(distances, axis=0))
            idx_new_centroid = np.argmin(np.sum(intra_cls_distances, axis=1))

            # print(np.array(intra_cls_distances[0]))
            # print('num per cluster', len(np.array(intra_cls_distances[0])))

            new_centroids.append(graphs_per_cls[idx_new_centroid])
        # print('##############################')
        return new_centroids

    cpdef bint are_centroids_equal(self, list cur_centroids, list old_centroids):
        for c_centroid, o_centroid in zip(cur_centroids, old_centroids):
            if c_centroid != o_centroid:
                return False
        return True

    cpdef double compute_SOD(self, list graphs, list centroids, int[::1] labels):
        cdef:
            double error = 0

        for k in range(self.n_clusters):
            cls_indices = np.where(np.array(labels) == k)[0]

            graphs_per_cls = [graphs[idx] for idx in cls_indices]
            # print(len(graphs_per_cls))
            intra_cls_distances = self.compute_distances(graphs_per_cls, centroids)
            # print(intra_cls_distances.base)
            # breakpoint()
            # print()
            # print(np.sum(intra_cls_distances, axis=0))
            # print(np.sum(intra_cls_distances))
            error += np.sum(intra_cls_distances)

        return error / self.n_clusters

    cpdef void fit(self, list graphs):
        cdef:
            list old_centroids
            double[:, ::1] distances

        self.centroids = self.init_centroids(graphs)

        for turn in range(self.max_iter):


            old_centroids = self.centroids

            distances = self.compute_distances(graphs,
                                               self.centroids)

            self.labels = self.find_closest_cluster(distances)

            self.error = self.compute_SOD(graphs, self.centroids, self.labels)

            self.centroids = self.update_centroids(graphs,
                                                   self.centroids,
                                                   self.labels)

            if self.are_centroids_equal(self.centroids, old_centroids):
                break

        self.error = self.compute_SOD(graphs, self.centroids, self.labels)
