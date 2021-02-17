cdef class LoaderTrainTestValSplit:

    cdef:
        str folder_dataset
        str __EXTENSION
        list X_train, y_train
        list X_test, y_test
        list X_val, y_val

    cdef list _init_splits(self, str filename)

    cpdef list load_train_split(self)

    cpdef list load_test_split(self)

    cpdef list load_val_split(self)
