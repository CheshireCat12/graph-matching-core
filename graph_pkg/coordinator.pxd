from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.edit_cost.edit_cost_AIDS cimport EditCostAIDS
from graph_pkg.edit_cost.edit_cost_letter cimport EditCostLetter
from graph_pkg.edit_cost.edit_cost_mutagenicity cimport EditCostMutagenicity
from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.loader.loader_base cimport LoaderBase
from graph_pkg.loader.loader_AIDS cimport LoaderAIDS
from graph_pkg.loader.loader_letter cimport LoaderLetter
from graph_pkg.loader.loader_mutagenicity cimport LoaderMutagenicity
from graph_pkg.loader.loader_train_test_val_split cimport LoaderTrainTestValSplit

cdef class Coordinator:

    cdef:
        readonly str dataset
        readonly tuple params_edit_cost
        readonly str folder_dataset
        readonly list graphs
        readonly list dataset_available

        EditCost edit_cost
        readonly GED ged
        LoaderBase loader
        LoaderTrainTestValSplit loader_split

    cdef int _init_system(self) except? -1

    cdef void _init_letter(self, str spec)

    cdef void _init_AIDS(self)

    cdef void _init_mutagenicity(self)

    cpdef tuple _split_dataset(self, list Xs, list ys)

    cpdef tuple train_split(self)

    cpdef tuple test_split(self)

    cpdef tuple val_split(self)
