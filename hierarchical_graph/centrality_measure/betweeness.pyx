import numpy as np
cimport numpy as np


cdef class Betweeness(CentralityMeasure):

    def __init__(self):
        super(Betweeness, self).__init__('betweeness')

    cpdef double[::1] calc_centrality_score(self, Graph graph):
        """
        Compute the betweeness centrality measure of the given graph
        Code inspired from : https://github.com/pvgupta24/Graph-Betweenness-Centrality/blob/master/serial.cpp
    
        :param graph: 
        :return: list[double] the betweeness score of all the nodes
        """
        cdef:
            int num_nodes = len(graph)
            double[::1] bw_centrality = np.zeros(num_nodes)
            dict predecessor = {} # defaultdict(list)

            double[::1] dependency
            int[::1] sigma
            int[::1] distance

            list stack = [] # use append(element) and pop()
            list queue = [] # use append(element) and pop(0)

            int i, v, w
            Edge edge
            Node node


        for s in range(num_nodes):

            dependency = np.zeros(num_nodes)
            sigma = np.zeros(num_nodes, dtype=np.int32)
            distance = np.ones(num_nodes, dtype=np.int32) * -1

            distance[s] = 0
            sigma[s] = 1

            queue.append(s)

            while queue:
                v = queue.pop(0)
                stack.append(v)

                for edge in graph.edges[v]:
                    if edge is None:
                        continue
                    # print(edge)
                    w = edge.idx_node_end

                    if distance[w] < 0:
                        queue.append(w)
                        distance[w] = distance[v] + 1

                    if distance[w] == distance[v] + 1:
                        sigma[w] += sigma[v]
                        predecessor.setdefault(w, []).append(v)
                        # predecessor.get(w, default=[]).append(v)

            while stack:
                w = stack.pop()

                for v in predecessor.get(w, []):
                    if sigma[w] != 0:
                        dependency[v] += ((sigma[v] * 1.0) / sigma[w]) * (1 + dependency[w])

                if w != s:
                    bw_centrality[w] += dependency[w] / 2

            for i in range(num_nodes):
                predecessor[i] = []

        return bw_centrality
