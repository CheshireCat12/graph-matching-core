cdef class Graph:
    """A class that is used to work with nodes and edges of a graph"""

    def __init__(self, str name, int num_nodes):
        self.name = name
        self.num_nodes = num_nodes
        self.nodes = [None] * num_nodes

    cpdef list get_nodes(self):
        return self.nodes

    cpdef int add_node(self, Node node) except? -1:
        assert node.idx < self.num_nodes, \
            f'The idx of the node {node.idx} exceed the number of nodes {self.num_nodes} authorized!'
        assert self.nodes[node.idx] is None, \
            f'The position {node.idx} is already used!'

        self.nodes[node.idx] = node

    # cpdef str get_label(self):
    #     return self.label
    #
    # cpdef int count_nodes(self):
    #     cdef:
    #         int counter = 0
    #         unsigned int idx = 0
    #
    #     for idx in range(self.num_nodes):
    #         if self.nodes[idx] != '\0':
    #             counter += 1
    #
    #     return counter
    #
    # cpdef void print_nodes(self):
    #     cdef Node tmp_node
    #
    #     for tmp_node in self.nodes:
    #         print(tmp_node.label)
    #
    #
    # cpdef void add_node(self, str lbl_node):
    #     """Add a new node to the graph."""
    #     self.nodes.append(Node(len(self.nodes) + 1, lbl_node))
    #
    # cpdef void add_edge(self):
    #     """Add a new edge to the graph."""

    def __repr__(self):
        eof = ",\n\t\t"
        return f'Graph: \n' \
               f'\tName: {self.name}\n' \
               f'\tNumber of nodes: {self.num_nodes}\n' \
               f'\tNodes: \n \t\t {eof.join(str(node) for node in self.nodes)}\n' \
               f'\tEdges: '


    def __len__(self):
        return self.num_nodes
