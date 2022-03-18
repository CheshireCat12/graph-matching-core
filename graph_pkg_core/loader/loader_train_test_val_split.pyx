from os.path import join
from glob import glob
from xmltodict import parse

cdef class LoaderTrainTestValSplit:
    """
    Load the split dataset.
    Take the folder dataset and retrieve the files with the split training, validation, and test sets.
    """

    def __init__(self, str folder_dataset):
        self.folder_dataset = folder_dataset

    cdef list _init_splits(self, str filename):
        cdef list data = []

        split_file = glob(join(self.folder_dataset, f'{filename}{EXTENSION_SPLITS}'))[0]

        with open(split_file) as file:
            split_text = "".join(file.readlines())

        parsed_data = parse(split_text)

        index = 'fingerprints'
        # if 'Mutagenicity' in self.folder_dataset:
        #     index = 'mutagenicity'

        splits = parsed_data['GraphCollection'][index]['print']

        for split in splits:
            data.append((split['@file'], int(split['@class'])))

        return data

    cpdef list load_train_split(self, str filename='train'):
        """
        Gather the training set.
        It returns a list of tuple containing the data and their corresponding labels

        :return: list((data, label))
        """
        return self._init_splits(filename)

    cpdef list load_test_split(self, str filename='test'):
        """
        Gather the test set.
        It returns a list of tuple containing the data and their corresponding labels

        :return: list((data, label))
        """
        return self._init_splits(filename)

    cpdef list load_val_split(self, str filename='validation'):
        """
        Gather the validation set.
        It returns a list of tuple containing the data and their corresponding labels

        :return: list((data, label))
        """
        return self._init_splits(filename)
