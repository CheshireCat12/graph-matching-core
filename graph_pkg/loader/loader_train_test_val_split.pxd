cdef class LoaderTrainTestValSplit:

    cdef:
        str folder_dataset
        str __EXTENSION
        list X_train, y_train
        list X_test, y_test
        list X_val, y_val

    cdef tuple _init_splits(self, str filename)

    cpdef tuple train_split(self)

    cpdef tuple test_split(self)

    cpdef tuple val_split(self)
