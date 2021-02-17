from graph_pkg.utils.constants cimport DEFAULT_FOLDERS_LABELS, DEFAULT_LABELS_TO_CODE


cdef class CoordinatorClassifier(Coordinator):

    def __init__(self,
                 str dataset,
                 tuple params_edit_cost,
                 str folder_dataset='',
                 str folder_labels=''):
        super().__init__(dataset, params_edit_cost, folder_dataset)
        self.folder_labels = folder_labels
        self.loader_split = LoaderTrainTestValSplit(self.folder_dataset)

    property folder_labels:
        def __get__(self):
            return self._folder_dataset
        def __set__(self, value):
            if value == '':
                self._folder_labels = DEFAULT_FOLDERS_LABELS[self.dataset]
            else:
                self._folder_labels = value

    property graph_filename_to_graph:
        def __get__(self):
            return {graph.filename: graph for graph in self.graphs}

    property lbl_to_code:
        def __get__(self):
            return DEFAULT_LABELS_TO_CODE[self.dataset]

    cpdef tuple _split_dataset(self, list data, bint conv_lbl_to_code=False):
        cdef:
            list graphs_split = []
            list labels_split = []
            Graph graph

        for graph_filename, label in data:
            graph = self.graph_filename_to_graph[graph_filename]

            if conv_lbl_to_code:
                label = self.lbl_to_code[label]

            graphs_split.append(graph)
            labels_split.append(label)

        return graphs_split, labels_split

    cpdef tuple train_split(self, bint conv_lbl_to_code=False):
        """
        Gather the training data.
        It returns a tuple of two lists. The first list contain the data.
        The second one contains the corresponding labels.
         
        :return: tuple(list, list)
        """
        cdef:
            list data

        data = self.loader_split.load_train_split()
        return self._split_dataset(data, conv_lbl_to_code)

    cpdef tuple test_split(self, bint conv_lbl_to_code=False):
        cdef:
            list data

        data = self.loader_split.load_test_split()
        return self._split_dataset(data, conv_lbl_to_code)

    cpdef tuple val_split(self, bint conv_lbl_to_code=False):
        cdef:
            list data

        data = self.loader_split.load_val_split()
        return self._split_dataset(data, conv_lbl_to_code)