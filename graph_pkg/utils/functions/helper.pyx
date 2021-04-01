import numpy as np


cpdef double calc_accuracy(int[::1] labels_ground_truth, int[::1] labels_predicted):
    correctly_classified = np.sum(np.array(labels_predicted) == np.array(labels_ground_truth))
    accuracy = 100 * (correctly_classified / len(labels_ground_truth))

    return accuracy