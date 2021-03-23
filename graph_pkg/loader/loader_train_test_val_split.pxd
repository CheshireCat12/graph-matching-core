cdef class LoaderTrainTestValSplit:

    cdef:
        str folder_dataset

    cdef list _init_splits(self, str filename)

    cpdef list load_train_split(self)

    cpdef list load_test_split(self)

    cpdef list load_val_split(self)
