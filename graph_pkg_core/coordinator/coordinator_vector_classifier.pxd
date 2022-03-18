from graph_pkg_core.coordinator.coordinator_vector cimport CoordinatorVector
from graph_pkg_core.loader.loader_train_test_val_split cimport LoaderTrainTestValSplit
from graph_pkg_core.graph.graph cimport Graph


cdef class CoordinatorVectorClassifier(CoordinatorVector):

    cdef:
        str _folder_labels
        readonly LoaderTrainTestValSplit loader_split

    cpdef tuple _split_dataset(self, list data)

    cpdef tuple train_split(self, str filename=*)

    cpdef tuple test_split(self, str filename=*)

    cpdef tuple val_split(self, str filename=*)
