from graph_pkg.utils.coordinator_gnn_embedding.coordinator_gnn_embedding cimport CoordinatorGNNEmbedding
from graph_pkg.loader.loader_train_test_val_split cimport LoaderTrainTestValSplit
from graph_pkg.graph.graph cimport Graph


cdef class CoordinatorGNNEmbeddingClassifier(CoordinatorGNNEmbedding):

    cdef:
        str _folder_labels
        readonly LoaderTrainTestValSplit loader_split

    cpdef tuple _split_dataset(self, list data)

    cpdef tuple train_split(self)

    cpdef tuple test_split(self)

    cpdef tuple val_split(self)
