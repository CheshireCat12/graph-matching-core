cdef class Coordinator:

    def __cinit__(self, str dataset, tuple params_edit_cost, str folder_dataset):
        self.dataset = dataset
        self.params_edit_cost = params_edit_cost
        self.folder_dataset = folder_dataset
        self._init_system()

    cdef int _init_system(self) except? -1:
        """
        
        :return: 
        """
        if 'letter' in self.dataset:
            self.dataset, spec = self.dataset.split('_')
            if spec is None or spec.lower() not in ['low', 'med', 'high']:
                raise ValueError(f'The dataset letter_{spec.lower()} is not available!')

            self._init_letter(spec.upper())
        elif self.dataset == 'AIDS':
            self._init_AIDS()
        elif self.dataset == 'mutagenicity':
            self._init_mutagenicity()
        else:
            raise ValueError(f'The dataset {self.dataset} is not available!')

        self.graphs = self.loader.load()
        self.ged = GED(self.edit_cost)
        self.loader_split = LoaderTrainTestValSplit(self.folder_dataset)

    cdef void _init_letter(self, str spec):
        self.loader = LoaderLetter(spec)
        self.edit_cost = EditCostLetter(*self.params_edit_cost)

    cdef void _init_AIDS(self):
        self.loader = LoaderAIDS()
        self.edit_cost = EditCostAIDS(*self.params_edit_cost)

    cdef void _init_mutagenicity(self):
        self.loader = LoaderMutagenicity()
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
        print(X_train)
        print("######")
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