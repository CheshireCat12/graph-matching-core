cdef class Coordinator:

    def __cinit__(self, str dataset, tuple params_edit_cost):
        self.dataset = dataset
        self.params_edit_cost = params_edit_cost
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

    cdef void _init_letter(self, str spec):
        self.loader = LoaderLetter(spec)
        self.edit_cost = EditCostLetter(*self.params_edit_cost)

    cdef void _init_AIDS(self):
        self.loader = LoaderAIDS()
        self.edit_cost = EditCostAIDS(*self.params_edit_cost)

    cdef void _init_mutagenicity(self):
        self.loader = LoaderMutagenicity()
        self.edit_cost = EditCostMutagenicity(*self.params_edit_cost)
