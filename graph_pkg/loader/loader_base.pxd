from graph_pkg.graph.edge cimport Edge
from graph_pkg.graph.graph cimport Graph
from graph_pkg.graph.node cimport Node
from graph_pkg.graph.label.label_base cimport LabelBase

cdef class LoaderBase:

    cdef:
        str _folder
        str __EXTENSION
        Graph _constructed_graph

    cpdef int _format_idx(self, str idx)

    cpdef LabelBase _formatted_lbl_node(self, attr)

    cpdef LabelBase _formatted_lbl_edge(self, attr)

    cpdef list load(self)

    cpdef void _construct_graph(self, str graph_filename, object parsed_data)
