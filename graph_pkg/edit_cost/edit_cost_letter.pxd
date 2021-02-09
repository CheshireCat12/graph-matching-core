from graph_pkg.edit_cost.edit_cost cimport EditCost
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node
from graph_pkg.edit_cost.metrics cimport letter_manhattan, letter_euclidean

ctypedef double (*metricptr)(double, double, double, double)

cdef class EditCostLetter(EditCost):

    cdef:
        double x1, y1, x2, y2, result
        int valence_source, valence_target

        str metric_name

        list metrics_available
        metricptr metric
