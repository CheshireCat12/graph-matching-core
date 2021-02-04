cdef class Node:
    cdef:
        unsigned int ID
        str label

    def __init__(self, int idx, str label):
        self.ID = idx
        self.label = label

cdef class Graph:
    """A class that is used to work with nodes and edges of a graph"""

    cdef:
        unsigned int ID
        str label
        unsigned int num_nodes
        int idx_node
        list nodes

    def __init__(self, str label, int num_nodes):
        self.label = label
        self.num_nodes = 5
        self.idx_node = 0


    cpdef str get_label(self):
        return self.label

    cpdef int count_nodes(self):
        cdef:
            int counter = 0
            unsigned int idx = 0

        for idx in range(self.num_nodes):
            if self.nodes[idx] != '\0':
                counter += 1

        return counter

    cpdef void print_nodes(self):
        cdef Node tmp_node

        for tmp_node in self.nodes:
            print(tmp_node.label)


    cpdef void add_node(self, str lbl_node):
        """Add a new node to the graph."""
        self.nodes.append(Node(len(self.nodes) + 1, lbl_node))

    cpdef void add_edge(self):
        """Add a new edge to the graph."""