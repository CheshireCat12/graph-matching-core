import os
from pathlib import Path

import numpy as np


cdef class Runner:

    def __init__(self, parameters):
        self.parameters = parameters
        self.folder = self.parameters.folder_results

        Path(self.folder).mkdir(parents=True, exist_ok=True)

    def run(self):
        raise NotImplementedError('Method run not implemented!')

    def save_predictions(self, int[::1] predictions, int[::1] labels_test, str name):
        filename = os.path.join(self.folder, name)

        with open(filename, 'wb') as f:
            np.save(f, labels_test)
            np.save(f, predictions)

    cpdef void save_stats(self, str message, str name, bint save_params=True):
        filename = os.path.join(self.folder, name)

        with open(filename, mode='a+') as fp:
            if save_params:
                fp.write(str(self.parameters))
            fp.write(f'{message}\n'
                     f'{"="*50}\n\n')

    cpdef void save_ged_distances(self, str name, double[:, ::1] distances):
        folder_dist = os.path.join(self.folder, 'distances')
        Path(folder_dist).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(folder_dist, f'ged_dist_{name}.npy')

        with open(filename, 'wb') as f:
            np.save(f, distances)