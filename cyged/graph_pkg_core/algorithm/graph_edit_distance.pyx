import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs as c_abs
from scipy.optimize import linear_sum_assignment

from cyged.graph_pkg_core.edit_cost.edit_cost cimport EditCost
from cyged.graph_pkg_core.graph.graph cimport Graph
from cyged.graph_pkg_core.graph.edge cimport Edge
from cyged.graph_pkg_core.graph.label.label_edge cimport LabelEdge

cdef class GED:
    """
    Compute the approximate graph edit distance with the linear assignment formulation.

    Attributes
    ----------
    edit_cost : EditCost

    Methods
    -------
    compute_edit_distance(graph_source, graph_target, heuristic)
    """

    def __init__(self, EditCost edit_cost):
        """

        :param edit_cost:
        """
        self.edit_cost = edit_cost

    cpdef double compute_edit_distance(self,
                                       Graph graph_source,
                                       Graph graph_target,
                                       bint heuristic=False):
        """
        Compute the graph edit distance
        
        Create the C matrix:
        Substitution | Deletion
        -----------------------
        Insertion    | None
        
        Then create the C star matrix to add the cost of the edges
        Once the C star matrix is created the node substitution is computed with the linear_sum_assignment function.
        
        :param graph_source: 
        :param graph_target: 
        :param heuristic: is set to true if the bigger graph has to be taken as the source graph
        :return: 
        """
        cdef:
            int i, j
            int[::1] phi
            double edit_cost = 0.


        self._init_graphs(graph_source, graph_target, heuristic)

        self._create_c_matrix()
        self._create_c_star_matrix()

        _, col_ind = linear_sum_assignment(self.C_star)
        self.phi = col_ind.astype(dtype=np.int32)

        edit_cost += self._compute_cost_node_edit(self.phi)
        edit_cost += self._compute_cost_edge_edit(self.phi)

        return edit_cost

    cdef void _init_graphs(self, Graph graph_source, Graph graph_target, bint heuristic):
        """if heuristic, the bigger graph is always source"""
        if heuristic and len(graph_source) < len(graph_target):
            self.graph_source = graph_target
            self.graph_target = graph_source
        else:
            self.graph_source = graph_source
            self.graph_target = graph_target

        self._n = len(self.graph_source)
        self._m = len(self.graph_target)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _create_c_matrix(self):
        cdef:

            int i, j
            double cost

        self.C = np.zeros((self._n + self._m, self._n + self._m),
                          dtype=np.float64)

        # Create substitute part
        for i in range(self._n):
            for j in range(self._m):
                cost = self.edit_cost.c_cost_substitute_node(self.graph_source.nodes[i],
                                                             self.graph_target.nodes[j])
                self.C[i][j] = cost

        # Create node deletion part
        self.C[0:self._n:1, self._m:self._n+self._m:1] = np.inf
        for i in range(self._n):
            j = self._m + i
            cost = self.edit_cost.c_cost_delete_node(self.graph_source.nodes[i])

            self.C[i][j] = cost

        # Create node insertion part
        self.C[self._n:self._n+self._m:1, 0:self._m:1] = np.inf
        for j in range(self._m):
            i = self._n + j
            cost = self.edit_cost.c_cost_insert_node(self.graph_target.nodes[j])

            self.C[i][j] = cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _create_c_star_matrix(self):
        cdef:
            int i, j
            double cost
            int[::1] out_degrees_source, out_degrees_target
            Edge edge_source, edge_target

        out_degrees_source = self.graph_source.degrees()
        out_degrees_target = self.graph_target.degrees()

        self.C_star = np.zeros((self._n + self._m, self._n + self._m),
                               dtype=np.float64)
        self.C_star[:, ::1] = self.C

        # Dirty fix
        # Create dumb edges to get the correct values from the edit_cost class
        edge_source = Edge(0, 1, LabelEdge(0))
        edge_target = Edge(0, 1, LabelEdge(0))

        # Update the substitution part
        for i in range(self._n):
            for j in range(self._m):
                cost = c_abs(out_degrees_source[i] - out_degrees_target[j]) * self.edit_cost.c_cost_insert_edge(edge_source)
                self.C_star[i][j] += cost

        # Update the deletion part
        for i in range(self._n):
            j = self._m + i
            cost = out_degrees_source[i] * self.edit_cost.c_cost_delete_edge(edge_source)

            self.C_star[i][j] += cost

        # Update the insertion part
        for j in range(self._m):
            i = self._n + j
            cost = out_degrees_target[j] * self.edit_cost.c_cost_insert_edge(edge_target)

            self.C_star[i][j] += cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_cost_node_edit(self, int[::1] phi):
        cdef:
            int i
            double cost = 0.

        for i in range(self._n + self._m):
            cost += self.C[i][phi[i]]

        return cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_cost_edge_edit(self, int[::1] phi):
        cdef:
            int i, j
            int phi_i, phi_j
            double cost
            double cost_edit = 0.
            Edge edge_source, edge_target

        for i in range(self._n + self._m):
            for j in range(i + 1, self._n + self._m):
                phi_i = phi[i]
                phi_j = phi[j]

                if self.graph_source.has_edge(i, j):
                    # check for edge substitution
                    if self.graph_target.has_edge(phi_i, phi_j):
                        edge_source = self.graph_source.get_edge_by_node_idx(i, j)
                        edge_target = self.graph_target.get_edge_by_node_idx(phi_i, phi_j)

                        cost = self.edit_cost.c_cost_substitute_edge(edge_source, edge_target)
                        cost_edit += cost

                    #check for edge deletion
                    else:
                        edge_source = self.graph_source.get_edge_by_node_idx(i, j)

                        cost = self.edit_cost.c_cost_delete_edge(edge_source)
                        cost_edit += cost

                else:
                    # check for edge insertion
                    if self.graph_target.has_edge(phi_i, phi_j):
                        edge_target = self.graph_target.get_edge_by_node_idx(phi_i, phi_j)

                        cost = self.edit_cost.c_cost_insert_edge(edge_target)
                        cost_edit += cost

        return cost_edit


    def __reduce__(self):
        d = dict()
        d['edit_cost'] = self.edit_cost

        return (rebuild, (d,))

def rebuild(data):
    cdef:
        GED ged
    ged = GED(data['edit_cost'])

    return ged
