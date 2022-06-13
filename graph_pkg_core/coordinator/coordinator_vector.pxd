from graph_pkg_core.edit_cost.edit_cost cimport EditCost
from graph_pkg_core.edit_cost.edit_cost_vector cimport EditCostVector
from graph_pkg_core.algorithm.graph_edit_distance cimport GED
from graph_pkg_core.loader.loader_vector cimport LoaderVector


cdef class CoordinatorVector:

    cdef:
        str _dataset
        str _folder_dataset
        readonly tuple params_edit_cost
        readonly list graphs

        readonly EditCost edit_cost
        readonly GED ged
        LoaderVector loader


    cdef int _init_system(self) except? -1

    cdef void _init_vector(self)
