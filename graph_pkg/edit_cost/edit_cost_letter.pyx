cdef class EditCostLetter(EditCost):

    def __cinit__(self):
        pass

    cpdef float cost_insert_node(self, Node node) except? -1:
        return 1.

    cdef float _compute_cost_insert_node(self, float x, float y) except? -1:
        raise NotImplementedError

    cpdef float cost_delete_node(self, Node node) except? -1:
        return 1.

    cpdef double cost_substitute_node(self, Node node1, Node node2) except? -1:
        self.x1, self.y1 = node1.label.get_attributes()
        self.x2, self.y2 = node2.label.get_attributes()

        self.result = self._compute_cost_substitute_node(self.x1, self.y1, self.x2, self.y2)

        return self.result

    cdef double _compute_cost_substitute_node(self, double x1, double y1, double x2, double y2) except? -1:
        raise NotImplementedError('substitute')

    cpdef float cost_insert_edge(self, Edge edge) except? -1:
        return 1.

    cpdef float cost_delete_edge(self, Edge edge) except? -1:
        return 1.

    cpdef float cost_substitute_edge(self, Edge edge1, Edge edge2) except? -1:
        return 1.
