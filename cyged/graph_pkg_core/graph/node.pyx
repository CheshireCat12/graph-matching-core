from cpython.object cimport Py_EQ

from cyged.graph_pkg_core.graph.label.label_base cimport LabelBase


cdef class Node:
    """
    Class used to represent the Node.
    A node contains an ID and a Label.

    Attributes
    ----------
    idx : int
    label : LabelBase

    Methods
    -------
    update_idx(new_idx)
    """

    def __init__(self, unsigned int idx, LabelBase label):
        """

        :param idx:
        :param label:
        """
        self.idx = idx
        self.label = label

    cdef void update_idx(self, unsigned int new_idx):
        """
        Update the index of the node with the new index.
        
        :param new_idx: 
        :return: 
        """
        self.idx = new_idx

    def __richcmp__(self, Node other, int op):
        assert isinstance(other, Node), f'The element {str(other)} is not a Node!'

        if op == Py_EQ:
            return self.idx == other.idx and \
                   self.label == other.label
        else:
            assert False


    def __repr__(self):
        return f'Node: {self.idx}, {str(self.label)}'

    def __hash__(self):
        return hash(self.idx)

