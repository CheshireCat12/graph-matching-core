from glob import glob
from xmltodict import parse


cdef class LoaderTrainTestValSplit:

    def __cinit__(self, str folder_dataset):
        self.folder_dataset = folder_dataset
        self.__EXTENSION = '.cxl'

    cdef tuple _init_splits(self, str filename):
        split_file = glob(f'{self.folder_dataset}{filename}{self.__EXTENSION}')[0]

        with open(split_file) as file:
            split_text = "".join(file.readlines())

        parsed_data = parse(split_text)
        index = 'fingerprints'
        if 'Mutagenicity' in self.folder_dataset:
            index = 'mutagenicity'
        splits = parsed_data['GraphCollection'][index]['print']
        X, y = [], []

        for split in splits:
            X.append(split['@file'])
            y.append(split['@class'])

        return X, y


    cpdef tuple train_split(self):
        return self._init_splits('train')

    cpdef tuple test_split(self):
        return self._init_splits('test')

    cpdef tuple val_split(self):
        return self._init_splits('validation')