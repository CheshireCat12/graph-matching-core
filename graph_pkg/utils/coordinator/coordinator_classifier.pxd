from graph_pkg.utils.coordinator.coordinator cimport Coordinator
from graph_pkg.loader.loader_train_test_val_split cimport LoaderTrainTestValSplit
from graph_pkg.graph.graph cimport Graph


cdef class CoordinatorClassifier(Coordinator):

    cdef:
        readonly str _folder_labels
        readonly LoaderTrainTestValSplit loader_split

    cpdef tuple _split_dataset(self, list data, bint conv_lbl_to_code=*)

    cpdef tuple train_split(self, bint conv_lbl_to_code=*)

    cpdef tuple test_split(self, bint conv_lbl_to_code=*)

    cpdef tuple val_split(self, bint conv_lbl_to_code=*)
