import copy
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport ceil
from time import time

from graph_pkg.utils.constants cimport PERCENT_HIERARCHY


cdef class HierarchicalGraphs:

    def __init__(self, list graphs, CentralityMeasure measure,
                 list percentage_hierarchy=PERCENT_HIERARCHY,
                 str deletion_strategy='compute_once', bint verbose=True,
                 bint augmented_random_graphs=False,
                 int num_sub_bunch=1):
        self.hierarchy = {}
        self.original_graphs = graphs
        self.percentage_hierarchy = percentage_hierarchy
        self.measure = measure
        self.deletion_strategy = deletion_strategy
        self.verbose = verbose

        if augmented_random_graphs:
            self.num_sub_bunch = num_sub_bunch
            self._create_hierarchy_random()

        else:
            self._create_hierarchy_of_graphs()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void _create_hierarchy_of_graphs(self):
        cdef:
            double percentage

        if self.verbose:
            print(f'\n** Create Graph Hierarchy with {self.measure.name} **')

        for percentage in self.percentage_hierarchy:

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


        mean_size_graph_original = 0
        mean_size_graph_reduced = 0

        for graph in graphs:
            mean_size_graph_original += len(graph)
            tmp_graph = copy.deepcopy(graph)

            # The graphs with less than 5 nodes aren't reduced!
            if len(graph) < 5:
                num_nodes_to_del = 0
            else:
                num_nodes_to_del = int(ceil((1.0 - percentage_remaining) * len(tmp_graph)))

            delete_strat(tmp_graph, num_nodes_to_del)

            reduced_graphs.append(tmp_graph)
            mean_size_graph_reduced += len(tmp_graph)

        if self.verbose:
            print(f'~~ Running time: {time() - start_time:.2f}s')
            print(f'~~     Mean size graphs original {mean_size_graph_original / len(graphs):.2f}')
            print(f'~~     Mean size graphs reduced {mean_size_graph_reduced / len(graphs):.2f}')

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void _create_hierarchy_random(self):
        """
        Create many random reduced graph space with various random node deletion

        :return:
        """
        cdef:
            double percentage

        if self.verbose:
            print(f'\n** Create Random Graph Hierarchy **')

        print(f'Create more random graph space #: {self.num_sub_bunch}')

        for percentage in self.percentage_hierarchy:
            if percentage == 1.0:
                self.hierarchy[percentage] = self._reduce_graphs(self.original_graphs,
                                                                 percentage)
            else:
                self.hierarchy[percentage] = [self._reduce_graphs(self.original_graphs,
                                                                  percentage)
                                              for _ in range(self.num_sub_bunch)]
