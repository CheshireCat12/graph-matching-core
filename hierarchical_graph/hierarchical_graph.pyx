import copy
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport ceil
from time import time

cdef class HierarchicalGraph:

    def __init__(self, list graphs, CentralityMeasure measure, sigma_js=None):
        self.level_graphs = [graphs]
        self.measure = measure
        self.sigma_js = sigma_js


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list create_hierarchy_percent(self, list graphs,
                                        double percentage_remaining=1.0,
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

        # copied_graphs = [copy.deepcopy(graph) for graph in graphs]

        if verbose:
            print('\n** Creating hierarchical representation **')
            start_time = time()

        for graph in graphs:
            tmp_graph = copy.deepcopy(graph)
            num_nodes_to_del = int(ceil((1.0 - percentage_remaining) * len(tmp_graph)))
            # print(num_nodes_to_del)
            # print(graph)
            # One node at a time and recompute the centrality score every turn
            for idx_tmp in range(num_nodes_to_del):
                centrality_score = np.asarray(self.measure.calc_centrality_score(tmp_graph))
                # print(centrality_score)

                idx_to_delete, *_ = np.where(centrality_score == centrality_score.min())

                idx_to_delete = idx_to_delete[:1]

                for idx_del in idx_to_delete:
                    tmp_graph.remove_node_by_idx(idx_del)

                # print('test')

            # print(f'Original {len(graph)}')
            # print(f'Reduced  {len(tmp_graph)}')
            # print(f'Percent: {len(tmp_graph) / len(graph)}')
            reduced_graphs.append(tmp_graph)

        if verbose:
            print(f'~~ Running time: {time() - start_time:.2f}s')

        return reduced_graphs

    cpdef void create_hierarchy_sigma(self, strategy='one_by_one'):
        cdef:
            int idx, idx_graph, current_level
            list graphs_original
            Graph graph


        current_level = 0
        total_number_nodes = [len(graph) for graph in self.level_graphs[0]]

        while current_level < 40:
            graphs_original = self.level_graphs[current_level]
            graphs_copy = [copy.deepcopy(graph) for graph in graphs_original]


            for idx_graph, graph in enumerate(graphs_copy):
                if len(graph) == 0:
                    continue

                centrality_score = np.asarray(self.measure.calc_centrality_score(graph))

                # Round the values when removing multiple nodes at once
                if strategy == 'multiple_by_one':
                    centrality_score = np.round(centrality_score, decimals=3)

                extra_info_nodes = f'Current nodes/Total nodes: {len(graph)}/{total_number_nodes[idx_graph]} <br>' \
                                   f'Percentage remaining: {(len(graph)/total_number_nodes[idx_graph])*100:.0f}%'

                self.sigma_js.save_to_sigma_with_score(graph,
                                                       centrality_score,
                                                       self.measure.name,
                                                       level=current_level,
                                                       extra_info=strategy,
                                                       extra_info_nodes=extra_info_nodes)

                idx_to_delete, *_ = np.where(centrality_score == centrality_score.min())

                if strategy == 'one_by_one':
                    node_idx_to_delete = idx_to_delete[:1]
                elif strategy == 'multiple_by_one':
                    node_idx_to_delete = idx_to_delete[::-1]
                else:
                    raise ValueError(f'Strategy: {strategy} not accepted!')
                # print(centrality_score)
                # print(np.round(centrality_score, decimals=3))
                # print(node_idx_to_delete)

                for idx in node_idx_to_delete:
                    graph.remove_node_by_idx(idx)

                # break

            current_level += 1

            self.level_graphs.append(graphs_copy)

        print(f'\n**  Hierarchy Created **')
