import os
import numpy as np


cpdef double calc_accuracy(int[::1] labels_ground_truth, int[::1] labels_predicted):
    """
    Compute the accuracy by comparing the ground truth labels with the predicted ones
    
    :param labels_ground_truth: 
    :param labels_predicted: 
    :return: double accuracy
    """
    correctly_classified = np.sum(np.array(labels_predicted) == np.array(labels_ground_truth))
    accuracy = 100 * (correctly_classified / len(labels_ground_truth))

    return accuracy

cpdef double calc_save_accuracy(int[::1] labels_ground_truth, int[::1] labels_predicted,
                                str name, str folder):
    """
    Compute the accuracy and save the predicted labels
    
    :param labels_ground_truth: 
    :param labels_predicted: 
    :param name: 
    :param folder: 
    :return: double accuracy
    """
    filename = os.path.join(folder, name)
    with open(filename, 'wb') as f:
        np.save(f, np.array(labels_ground_truth))
        np.save(f, np.array(labels_predicted))

    accuracy = calc_accuracy(labels_ground_truth, labels_predicted)

    return accuracy