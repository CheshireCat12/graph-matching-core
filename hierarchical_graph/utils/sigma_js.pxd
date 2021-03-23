from graph_pkg.graph.graph cimport Graph
from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.node cimport Node

cdef class SigmaJS:

    cdef:
        str dataset, folder_results
        bint save_html, save_json