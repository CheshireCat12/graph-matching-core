cdef class CoordinatorVectorClassifier(CoordinatorVector):
    """
    Coordinator Classifier is a subclass of the Coordinator.
    It is used to split the dataset into the training, validation, and test set.

    Attributes
    ----------
    folder_labels : str
    loader_split : LoaderTrainTestValSplit
    graph_filename_to_graph : dict
    lbl_to_code : dict

    Methods
    -------
    train_split(conv_lbl_to_code=False)
    test_split(conv_lbl_to_code=False)
    val_split(conv_lbl_to_code=False)
    """

    def __init__(self,
                 str dataset_name,
                 tuple params_edit_cost,
                 str folder_dataset,
                 str folder_labels=None,
                 bint verbose = False):
        """

        Args:
            dataset_name:
            params_edit_cost:
            folder_dataset:
            folder_labels:
            verbose:
        """
        super().__init__(dataset_name, params_edit_cost, folder_dataset, verbose)
        self.folder_labels = folder_labels
        self.loader_split = LoaderTrainTestValSplit(self.folder_dataset)

    @property
    def folder_labels(self):
        return self._folder_labels

    @property
    def graph_filename_to_graph(self):
        return {graph.filename: graph for graph in self.graphs}

    @folder_labels.setter
    def folder_labels(self, value):
        if value is None:
            self._folder_labels = self.folder_dataset
        else:
            self._folder_labels = value

    cpdef tuple _split_dataset(self, list data):
        cdef:
            list graphs_split = []
            list labels_split = []
            Graph graph

        for graph_filename, label in data:
            graph = self.graph_filename_to_graph[graph_filename]
            graphs_split.append(graph)
            labels_split.append(label)

        return graphs_split, labels_split

    cpdef tuple train_split(self, str filename='train'):
        """
        Gather the training data.
        It returns a tuple of two lists. The first list contain the data.
        The second one contains the corresponding labels.
    
        :return: tuple(list, list)
        """
        cdef:
            list data

        data = self.loader_split.load_train_split(filename)
        return self._split_dataset(data)

    cpdef tuple val_split(self, str filename='validation'):
        """
        Gather the validation data.
        It returns a tuple of two lists. The first list contain the data.
        The second one contains the corresponding labels.
    
        :return: tuple(list, list)
        """
        cdef:
            list data

        data = self.loader_split.load_val_split(filename)
        return self._split_dataset(data)

    cpdef tuple test_split(self, str filename='test'):
        """
        Gather the test data.
        It returns a tuple of two lists. The first list contain the data.
        The second one contains the corresponding labels.
    
        :return: tuple(list, list)
        """
        cdef:
            list data

        data = self.loader_split.load_test_split()
        return self._split_dataset(data)

    def __repr__(self):
        return super().__repr__() + f'Folder Labels: {self.folder_labels}'

    def __str__(self):
        indent = '   '
        str_parent = super().__str__().split('\n')
        str_parent[0] = f'{indent}Parameters CoordinatorClassifier:'
        return '\n'.join(str_parent) + f'{indent * 2}Folder Labels : {self.folder_labels}\n'
