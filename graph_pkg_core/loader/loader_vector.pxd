from graph_pkg_core.graph.edge cimport Edge
from graph_pkg_core.graph.graph cimport Graph
from graph_pkg_core.graph.node cimport Node
from graph_pkg_core.graph.label.label_base cimport LabelBase
from graph_pkg_core.graph.label.label_node_vector cimport LabelNodeVector
from graph_pkg_core.graph.label.label_edge cimport LabelEdge
from graph_pkg_core.utils.constants cimport EXTENSION_GRAPHML

cdef class LoaderVector:
    cdef:
        bint _verbose
        str _folder
        str __EXTENSION
        Graph _constructed_graph

    cpdef int _format_idx(self, str idx)

    cpdef int _gr_idx_from_filename(self, str graph_folder)

    cpdef LabelBase _formatted_lbl_node(self, attr)

    cpdef LabelBase _formatted_lbl_edge(self, attr)

    cpdef list load(self)

    cpdef void _construct_graph(self, str graph_filename, object parsed_data)
