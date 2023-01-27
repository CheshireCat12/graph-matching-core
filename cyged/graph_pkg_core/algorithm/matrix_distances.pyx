cimport cython
import numpy as np
cimport numpy as np
from progress.bar import Bar
from multiprocessing import Pool

from cyged.graph_pkg_core.algorithm.graph_edit_distance cimport GED
from cyged.graph_pkg_core.graph.graph cimport Graph

import psutil


cdef class MatrixDistances:
    """
    Compute the graph edit distance between the two lists of graphs.
    """

    def __init__(self, GED ged, bint parallel=False, verbose=False):
        """
        :param ged: the graph edit distance class
        :param parallel: Bool - choose to use the serial or the parallel computation
        """
        self.parallel = parallel
        self.ged = ged
        self.verbose = verbose

    cpdef double[:, ::1] calc_matrix_distances(self,
                                               list graphs_train,
                                               list graphs_test,
                                               bint heuristic=False,
                                               int num_cores=-1):
        """
        Compute all the distances between the graphs in the lists given
        in parameter.
        The heuristic of the graph edit distance is activated!
        Therefore, the order of the graph given to the ged does not matter.

        If the parallel is activated, the distances are computed using the maximum
        amount of CPUs available.

        :param graphs_train: list of graphs
        :param graphs_test: list of graphs
        :param heuristic: bool - if the biggest is taken as source
        :return: distances between the graphs in the given lists
        """
        if self.parallel:
            return self._parallel_calc_matrix_distances(graphs_train,
                                                        graphs_test,
                                                        heuristic,
                                                        num_cores)
        else:
            return self._serial_calc_matrix_distances(graphs_train,
                                                      graphs_test,
                                                      heuristic)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:, ::1] _serial_calc_matrix_distances(self,
                                                       list graphs_train,
                                                       list graphs_test,
                                                       bint heuristic=False):
        cdef:
            int i, j, n
            double edit_cost
            double[:, ::1] distances
            Graph graph_source, graph_target

        if self.verbose:
            print('~~ Serial Computation\n')

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:, ::1] _parallel_calc_matrix_distances(self,
                                                         list graphs_train,
                                                         list graphs_test,
                                                         bint heuristic=False,
                                                         int num_cores=-1):
        cdef:
            Graph graph_source, graph_target

        n, m = len(graphs_train), len(graphs_test)

        prods = []
        for graph_source in graphs_train:
            for graph_target in graphs_test:
                prods.append((graph_source, graph_target))

        results = self.do_parallel_computation(prods, heuristic, num_cores)

        distances = np.array(results).reshape((n, m))
        return distances

    cpdef double[::1] do_parallel_computation(self,
                                    list prods,
                                    bint heuristic=False,
                                    int num_cores=-1):

        num_cores = psutil.cpu_count() if num_cores <= 0 else num_cores

        if self.verbose:
            print(f'~~ Parallel Computation')
            print(f'~~ Number of cores: {num_cores}')

        pool = Pool(num_cores)
        with pool as p:
            results = p.starmap(self._helper_parallel,
                                [(graph_train, graph_test, heuristic)
                                 for graph_train, graph_test in prods])

        distances = np.array(results)
        return distances

    cpdef double _helper_parallel(self,
                                  Graph graph_train,
                                  Graph graph_test,
                                  bint heuristic=False):
        dist = self.ged.compute_edit_distance(graph_train, graph_test, heuristic)

        return dist