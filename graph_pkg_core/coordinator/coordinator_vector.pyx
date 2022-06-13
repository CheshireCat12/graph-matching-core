"""
@author: Anthony Gillioz

This file contains the coordinator class. It is used to load the graphs given the name of the dataset.
The EditCost is also loaded automatically given the name of the dataset.
"""

cdef class CoordinatorVector:
    """
    Coordinator class coordinate the datasets with their corresponding loaders and edit costs.

    Attributes
    ----------
    dataset : str
    folder_dataset : str
    params_edit_cost : tuple
    graphs : list
    ged : GED

    """

    def __init__(self,
                 str dataset_name,
                 tuple params_edit_cost,
                 str folder_dataset='',
                 bint verbose=False):
        self.dataset = dataset_name
        self.params_edit_cost = params_edit_cost
        self.folder_dataset = folder_dataset
        self._init_system()

        print('\n** Coordinator Vector Loaded **')
        if verbose:
            print(self)

    @property
    def dataset(self):
        return self._dataset

    @property
    def folder_dataset(self):
        return self._folder_dataset

    @dataset.setter
    def dataset(self, str value):
        self._dataset = value

    @folder_dataset.setter
    def folder_dataset(self, str value):
        self._folder_dataset = value

    cdef int _init_system(self) except? -1:
        """
        Init the correct value for the given dataset.

        :return:
        """
        self._init_vector()

        self.graphs = self.loader.load()
        self.ged = GED(self.edit_cost)

    cdef void _init_vector(self):
        self.loader = LoaderVector(self.folder_dataset)
        self.edit_cost = EditCostVector(*self.params_edit_cost)

    def __repr__(self):
        return f'Coordinator - Dataset: {self.dataset}; ' \
               f'Parameters Cost: {self.edit_cost}; ' \
               f'Dataset Folder: {self.folder_dataset}; '

    def __str__(self):
        indent_ = '   '
        split_char = ';\n'
        edit_cst = f'\n{indent_ * 3}' + f'\n{indent_ * 3}'.join(str(self.edit_cost).split(';\n'))

        return f'{indent_}Parameters Coordinator:\n' \
               f'{indent_ * 2}Dataset: {self.dataset}\n' \
               f'{indent_ * 2}Parameters Cost: {edit_cst}\n' \
               f'{indent_ * 2}Folder dataset: {self.folder_dataset}\n'
