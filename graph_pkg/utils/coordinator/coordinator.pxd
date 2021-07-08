from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.edit_cost.edit_cost_AIDS cimport EditCostAIDS
from graph_pkg.edit_cost.edit_cost_letter cimport EditCostLetter
from graph_pkg.edit_cost.edit_cost_mutagenicity cimport EditCostMutagenicity
from graph_pkg.edit_cost.edit_cost_NCI1 cimport EditCostNCI1
from graph_pkg.edit_cost.edit_cost_proteins_tu cimport EditCostProteinsTU
from graph_pkg.edit_cost.edit_cost_enzymes cimport EditCostEnzymes
from graph_pkg.edit_cost.edit_cost_collab cimport EditCostCollab
from graph_pkg.edit_cost.edit_cost_reddit_binary cimport EditCostRedditBinary
from graph_pkg.algorithm.graph_edit_distance cimport GED
from graph_pkg.loader.loader_base cimport LoaderBase
from graph_pkg.loader.loader_AIDS cimport LoaderAIDS
from graph_pkg.loader.loader_letter cimport LoaderLetter
from graph_pkg.loader.loader_mutagenicity cimport LoaderMutagenicity
from graph_pkg.loader.loader_NCI1 cimport LoaderNCI1
from graph_pkg.loader.loader_proteins_tu cimport LoaderProteinsTU
from graph_pkg.loader.loader_enzymes cimport LoaderEnzymes
from graph_pkg.loader.loader_collab cimport LoaderCollab
from graph_pkg.loader.loader_reddit_binary cimport LoaderRedditBinary

cdef class Coordinator:

    cdef:
        str _dataset
        str _folder_dataset
        readonly tuple params_edit_cost
        readonly list graphs

        readonly EditCost edit_cost
        readonly GED ged
        LoaderBase loader

    cdef int _init_system(self) except? -1

    cdef void _init_letter(self)

    cdef void _init_AIDS(self)

    cdef void _init_mutagenicity(self)

    cdef void _init_NCI1(self)

    cdef void _init_proteins_tu(self)

    cdef void _init_enzymes(self)

    cdef void _init_collab(self)

    cdef void _init_reddit_binary(self)
