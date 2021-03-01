import copy
import numpy as np
cimport numpy as np


cdef class HierarchicalGraph:

    def __init__(self, list graphs, CentralityMeasure measure, SigmaJS sigma_js):
        self.level_graphs = graphs
        self.measure = measure
        self.sigma_js = sigma_js

    cpdef void create_hierarchy(self):
        cdef:
            int idx, current_level
            double[::1] centrality_score
            list graphs_original
            Graph graph

        current_level = 0

        graphs_original = self.level_graphs[current_level]
        graphs_copy = [copy.deepcopy(graph) for graph in graphs_original]

        for graph in graphs_copy:
            if len(graph) == 0:
                continue

            centrality_score = np.asarray(self.measure.calc_centrality_score(graph))
            self.sigma_js.save_to_sigma_with_score(graph, centrality_score, level=current_level)

            idx_to_delete = np.where(centrality_score == centrality_score.min())

            for idx in idx_to_delete:
                graph.remove_node_by_idx(idx)

        self.level_graphs.append(graphs_copy)
