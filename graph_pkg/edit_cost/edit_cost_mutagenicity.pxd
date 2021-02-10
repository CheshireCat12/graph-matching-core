from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node
from graph_pkg.edit_cost.metrics cimport dirac_mutagenicity

ctypedef double (*metricptr)(int, int)

cdef class EditCostMutagenicity(EditCost):

    cdef:
        int chem_source, chem_target
        int valence_source, valence_target

        list metrics_available
        metricptr metric
