from cpython.object cimport Py_EQ

cdef class Node:

    def __cinit__(self, unsigned int idx, LabelBase label):
        self.idx = idx
        self.label = label

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

