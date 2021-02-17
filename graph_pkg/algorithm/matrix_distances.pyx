cimport cython
import numpy as np
cimport numpy as np
from progress.bar import Bar

cdef class MatrixDistances:
    """
    Compute the graph edit distance between the two lists of graphs.
    """

    def __init__(self, GED ged):
        self.ged = ged

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:, ::1] calc_matrix_distances(self,
                                               list graphs_train,
                                               list graphs_test,
                                               bint heuristic=False):
        """
        Compute all the distances between the graphs in the lists given 
        in parameter.
        The heuristic of the graph edit distance is activated!
        Therefore, the order of the graph given to the ged does not matter.
        
        
        :param graphs_train: list of graphs
        :param graphs_test: list of graphs
        :param heuristic: bool - if the biggest is taken as source
        :return: distances between the graphs in the given lists
        """
        cdef:
            int i, j, n
            double edit_cost
            double[:, ::1] distances
            Graph graph_source, graph_target

        n = len(graphs_train)
        m = len(graphs_test)
        distances = np.full((n, m), fill_value=np.inf, dtype=np.float64)

        bar = Bar('Processing', max=n)
        for i in range(n):
            graph_source = graphs_train[i]

            for j in range(m):
                graph_target = graphs_test[j]

                edit_cost = self.ged.compute_edit_distance(graph_source,
                                                           graph_target,
                                                           heuristic=heuristic)
                distances[i][j] = edit_cost

            bar.next()
        bar.finish()
        return distances
