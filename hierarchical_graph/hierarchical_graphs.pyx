import copy
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport ceil
from time import time

from graph_pkg.utils.constants cimport PERCENT_HIERARCHY


cdef class HierarchicalGraphs:


    def __init__(self, list graphs, CentralityMeasure measure,
                 str deletion_strategy='compute_once', bint verbose=True):
        self.hierarchy = {}
        self.original_graphs = graphs
        self.measure = measure
        self.deletion_strategy = deletion_strategy
        self.verbose = verbose

        self._create_hierarchy_of_graphs()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void _create_hierarchy_of_graphs(self):
        cdef:
            double percentage

        if self.verbose:
            print(f'\n** Create Graph Hierarchy with {self.measure.name} **')

        for percentage in PERCENT_HIERARCHY:

            self.hierarchy[percentage] = self._reduce_graphs(self.original_graphs,
                                                             percentage)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list _reduce_graphs(self, list graphs, double percentage_remaining=1.0):
        """
        Reduce the size of the graph until it remains the given percentage of nodes from the original graphs.
        
        :param graphs: 
        :param percentage_remaining: 
        :return: 
        """
        cdef:
            int idx_tmp, idx_del, num_nodes_to_del
            list reduced_graphs = []
            Graph graph, tmp_graph

        if self.verbose:
            print(f'~~ Create graph with {percentage_remaining*100:.0f}% of remaining nodes')
            start_time = time()

        if self.deletion_strategy == 'recomputing':
            delete_strat = self._update_graph_recomputing
        elif self.deletion_strategy == 'compute_once':
            delete_strat = self._update_graph_compute_once


        for graph in graphs:
            tmp_graph = copy.deepcopy(graph)

            # The graphs with less than 5 nodes aren't reduced!
            if len(graph) < 5:
                num_nodes_to_del = 0
            else:
                num_nodes_to_del = int(ceil((1.0 - percentage_remaining) * len(tmp_graph)))

            delete_strat(tmp_graph, num_nodes_to_del)

            reduced_graphs.append(tmp_graph)

        if self.verbose:
            print(f'~~ Running time: {time() - start_time:.2f}s')

        return reduced_graphs

    cpdef void _update_graph_compute_once(self, Graph graph, int num_nodes_to_del):
        """
        Delete one node at a time, compute the centrality once
        ! Caution the deletion is an inplace operation !

        :param graph: 
        :param num_nodes_to_del: 
        :return: 
        """
        centrality_score = np.asarray(self.measure.calc_centrality_score(graph))

        idx_sorted = np.argpartition(centrality_score, num_nodes_to_del)
        idx_to_delete = idx_sorted[:num_nodes_to_del]
        idx_to_delete_sorted = np.sort(idx_to_delete, kind='stable')[::-1]

        for idx_del in idx_to_delete_sorted:
            graph.remove_node_by_idx(idx_del)

    cpdef void _update_graph_recomputing(self, Graph graph, int num_nodes_to_del):
        """
        Delete one node at a time and recompute the centrality score every turn.
        ! Caution the deletion is an inplace operation !
        
        :param graph: 
        :param num_nodes_to_del: 
        :return: 
        """
        for idx_tmp in range(num_nodes_to_del):
            centrality_score = np.asarray(self.measure.calc_centrality_score(graph))

            idx_to_delete, *_ = np.where(centrality_score == centrality_score.min())

            idx_del = idx_to_delete[0]

            graph.remove_node_by_idx(idx_del)