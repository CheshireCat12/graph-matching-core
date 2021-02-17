from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.edit_cost.edit_cost_AIDS cimport EditCostAIDS
from graph_pkg.edit_cost.edit_cost_letter cimport EditCostLetter
from graph_pkg.edit_cost.edit_cost_mutagenicity cimport EditCostMutagenicity
from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.loader.loader_base cimport LoaderBase
from graph_pkg.loader.loader_AIDS cimport LoaderAIDS
from graph_pkg.loader.loader_letter cimport LoaderLetter
from graph_pkg.loader.loader_mutagenicity cimport LoaderMutagenicity

cdef class Coordinator:

    cdef:
        str _dataset
        str _folder_dataset
        readonly tuple params_edit_cost
        readonly list graphs
        readonly list dataset_available

        EditCost edit_cost
        readonly GED ged
        LoaderBase loader

    cdef int _init_system(self) except? -1

    cdef void _init_letter(self)

    cdef void _init_AIDS(self)

    cdef void _init_mutagenicity(self)
