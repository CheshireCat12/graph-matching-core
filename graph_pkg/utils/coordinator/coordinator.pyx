from graph_pkg.utils.constants cimport DEFAULT_FOLDERS, DEFAULT_FOLDERS_LABELS

cdef class Coordinator:

    def __init__(self, str dataset,
                 tuple params_edit_cost,
                 str folder_dataset='',
                 str folder_labels=''):
        self.dataset = dataset
        self.params_edit_cost = params_edit_cost
        self.folder_dataset = folder_dataset
        self.folder_labels = folder_labels
        self._init_system()

    property dataset:
        def __get__(self):
            return self._dataset
        def __set__(self, str value):
            assert value in DEFAULT_FOLDERS, f'The dataset {value} is not available!'
            self._dataset = value

    property folder_dataset:
        def __get__(self):
            return self._folder_dataset
        def __set__(self, value):
            if value == '':
                self._folder_dataset = DEFAULT_FOLDERS[self.dataset]
            else:
                self._folder_dataset = value

    property folder_labels:
        def __get__(self):
            return self._folder_dataset
        def __set__(self, value):
            if value == '':
                self._folder_labels = DEFAULT_FOLDERS_LABELS[self.dataset]
            else:
                self._folder_labels = value


    cdef int _init_system(self) except? -1:
        """
        
        :return: 
        """
        if self.dataset == 'letter':
            self._init_letter()
        elif self.dataset == 'AIDS':
            self._init_AIDS()
        elif self.dataset == 'mutagenicity':
            self._init_mutagenicity()
        else:
            raise ValueError(f'The dataset {self.dataset} is not available!')

        self.graphs = self.loader.load()
        self.ged = GED(self.edit_cost)
        self.loader_split = LoaderTrainTestValSplit(self.folder_dataset)

    cdef void _init_letter(self):
        self.loader = LoaderLetter(self.folder_dataset)
        self.edit_cost = EditCostLetter(*self.params_edit_cost)

    cdef void _init_AIDS(self):
        self.loader = LoaderAIDS(self.folder_dataset)
        self.edit_cost = EditCostAIDS(*self.params_edit_cost)

    cdef void _init_mutagenicity(self):
        self.loader = LoaderMutagenicity(self.folder_dataset)
        self.edit_cost = EditCostMutagenicity(*self.params_edit_cost)

    cpdef tuple _split_dataset(self, list Xs, list ys):
        cdef:
            list Xs_split
            dict ys_split

        Xs_split = [graph for graph in self.graphs if graph.filename in Xs]
        ys_split = {gr_name: class_ for gr_name, class_ in zip(Xs, ys)}

        return Xs_split, ys_split

    cpdef tuple train_split(self):
        cdef:
            list X_train, y_train

        X_train, y_train = self.loader_split.train_split()
        # print(X_train)
        # print("######")
        return self._split_dataset(X_train, y_train)

    cpdef tuple test_split(self):
        cdef:
            list X_test, y_test

        X_test, y_test = self.loader_split.test_split()
        return self._split_dataset(X_test, y_test)

    cpdef tuple val_split(self):
        cdef:
            list X_val, y_val

        X_val, y_val = self.loader_split.val_split()
        return self._split_dataset(X_val, y_val)