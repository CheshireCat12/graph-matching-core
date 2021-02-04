cdef class Edge:

    cdef:
        unsigned int idx
        readonly unsigned int start_node_idx
        readonly unsigned int end_node_idx
