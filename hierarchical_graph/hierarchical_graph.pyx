import copy
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport ceil
from time import time

cdef class HierarchicalGraph:


    def __init__(self, list graphs, CentralityMeasure measure): # , sigma_js=None):
        self.level_graphs = [graphs]
        self.measure = measure
        # self.sigma_js = sigma_js


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list create_hierarchy_percent(self, list graphs,
                                        double percentage_remaining=1.0,
                                        str deletion_strategy="recomputing",
                                        bint verbose=True):
        """
        
        :param graphs: 
        :param percentage_remaining: 
        :return: 
        """
        cdef:
            int idx_tmp, idx_del, num_nodes_to_del
            list reduced_graphs = []
            Graph graph, tmp_graph

        if verbose:
            print('\n** Creating hierarchical representation **')
            start_time = time()

        if deletion_strategy == 'recomputing':
            delete_strat = self._update_graph_recomputing
        elif deletion_strategy == 'compute_once':
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

        if verbose:
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

        # if self.sigma_js is not None:
        #     tmp_idx = np.sort(idx_sorted[num_nodes_to_del:])
        #     # print(centrality_score[tmp_idx])
        #     self._save_graph_to_js(graph, num_nodes_to_del, centrality_score[tmp_idx])

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

    # cpdef void _save_graph_to_js(self, Graph graph, int num_nodes_deleted, double[::1] centrality_score):
    #     original_size = len(graph) + num_nodes_deleted
    #     current_size = len(graph)
    #     percentage_size = current_size / original_size
    #     rounded_percentage_size = ceil(percentage_size * 10) / 10
    #     extra_info = f'percentage_{rounded_percentage_size}'
    #
    #     extra_info_nodes = f'Current nodes/Total nodes: {current_size}/{original_size} <br>' \
    #                        f'Percentage remaining: {rounded_percentage_size * 100:.0f}%'
    #
    #     # print(centrality_score)
    #     self.sigma_js.save_to_sigma_with_score(graph,
    #                                            centrality_score,
    #                                            self.measure.name,
    #                                            level=rounded_percentage_size*100,
    #                                            extra_info=extra_info,
    #                                            extra_info_nodes=extra_info_nodes)