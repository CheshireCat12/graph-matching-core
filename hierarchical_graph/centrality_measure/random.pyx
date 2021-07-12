import numpy as np
cimport numpy as np


cdef class Random(CentralityMeasure):

    def __init__(self, int seed=19):
        super(Random, self).__init__('random')
        np.random.seed(seed)

    cpdef double[::1] calc_centrality_score(self, Graph graph):
        """
        Return a random number between 0,1 as a centrality score
        (Sanity check: to verify if our method is not flawed)

        :param graph: 
        :return: np.array[double] - PageRank score of the nodes
        """
        cdef:
            int num_nodes = len(graph)
            double[::1] random_centrality = np.random.random(num_nodes)

        return random_centrality


