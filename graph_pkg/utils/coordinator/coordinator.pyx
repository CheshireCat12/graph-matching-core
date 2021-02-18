from graph_pkg.utils.constants cimport DEFAULT_FOLDERS

cdef class Coordinator:

    def __init__(self,
                 str dataset,
                 tuple params_edit_cost,
                 str folder_dataset=''):
        self.dataset = dataset
        self.params_edit_cost = params_edit_cost
        self.folder_dataset = folder_dataset
        self._init_system()

        print('** Coordinator Loaded **')
        print(self)

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

    cdef void _init_letter(self):
        self.loader = LoaderLetter(self.folder_dataset)
        self.edit_cost = EditCostLetter(*self.params_edit_cost)

    cdef void _init_AIDS(self):
        self.loader = LoaderAIDS(self.folder_dataset)
        self.edit_cost = EditCostAIDS(*self.params_edit_cost)

    cdef void _init_mutagenicity(self):
        self.loader = LoaderMutagenicity(self.folder_dataset)
        self.edit_cost = EditCostMutagenicity(*self.params_edit_cost)

    def __repr__(self):
        return f'Coordinator - Dataset: {self.dataset}; ' \
               f'Parameters Cost: {self.edit_cost}; ' \
               f'Dataset Folder: {self.folder_dataset}; '

    def __str__(self):
        indent_ = '   '
        split_char = ';\n'
        edit_cst = f'\n{indent_*3}' +  f'\n{indent_*3}'.join(str(self.edit_cost).split(';\n'))

        return f'{indent_}Parameters Coordinator:\n' \
               f'{indent_*2}Dataset: {self.dataset}\n' \
               f'{indent_*2}Parameters Cost: {edit_cst}\n' \
               f'{indent_*2}Folder dataset: {self.folder_dataset}\n'
