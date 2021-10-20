from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.edit_cost.edit_cost_gnn_embedding cimport EditCostGNNEmbedding
from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.loader_gnn_embedding.loader_gnn_embedding_base cimport LoaderGNNEmbeddingBase as LoaderGNNEmbedding

cdef class CoordinatorGNNEmbedding:

    cdef:
        str _dataset
        str _folder_dataset
        readonly tuple params_edit_cost
        readonly list graphs

        readonly EditCost edit_cost
        readonly GED ged
        LoaderGNNEmbedding loader

    cdef int _init_system(self) except? -1

    cdef void _init_gnn_embedding(self)
